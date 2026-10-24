"""Regression coverage for the wide B-roll review, using only synthetic media."""

import subprocess
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest
import synth_broll
from test_assemble_picture import heic_quadrants
from test_planner_broll import catalog_item, phrase_words

from yap_studio import ffmpeg, paths, receipt
from yap_studio.faces.model import Box, FaceBox, FacesData, VideoIdentity
from yap_studio.stages import assemble_picture as picture
from yap_studio.stages.broll_framing import wide_insert
from yap_studio.stages.timeline import ResolvedElement


@pytest.mark.parametrize('fit', ['cover', 'contain'])
@pytest.mark.parametrize('size', ['180x320', '240x240'])
def test_vertical_split_preserves_fit_and_motion(tmp_path, fit, size):
    source = picture._source(synth_broll.png(tmp_path / 'photo.png', size=size), [], [],
                             owner='broll-split')
    item = ResolvedElement('broll-split', 'insert', 'broll', 0, 6, 'front',
        {'media': {'fit': fit}, 'motion': 'push', 'push_scale': 1.08, 'push_direction': 'in'})
    graph = picture._Graph()
    picture._insert(graph, source, item, size=(1080, 960))
    filters = ';'.join(graph.filters)
    assert ('force_original_aspect_ratio=decrease' in filters) == (fit == 'contain')
    assert 'brightness=' not in filters
    assert 'eval=frame' in filters


def test_wide_blur_bounds_use_delivery_canvas(tmp_path):
    source = synth_broll.png(tmp_path / 'photo.png')
    item = ResolvedElement('broll-blur', 'insert', 'broll', 0, 6, 'front',
        {'media': {'path': source.name, 'sha256': receipt.sha256_file(source),
                   'blur': [[1300, 300, 200, 200]]}})
    timeline = SimpleNamespace(elements=(item,))
    assert picture._insert_inputs(tmp_path, timeline, [], [], size=(1920, 1080))
    invalid = replace(item, payload={'media': {**item.payload['media'],
                                              'blur': [[300, 1300, 200, 200]]}})
    with pytest.raises(picture.PictureError, match='inside canvas'):
        picture._insert_inputs(tmp_path, SimpleNamespace(elements=(invalid,)), [], [],
                               size=(1920, 1080))


@pytest.mark.parametrize('placement', ['overlay', 'split', 'green_screen'])
@pytest.mark.parametrize('kind', ['image', 'video'])
def test_wide_suggestions_plan_supported_placements(tmp_path, placement, kind):
    from test_assemble_picture import write_document

    from yap_studio.planner.broll import plan_broll, suggest_broll

    item = catalog_item(tmp_path, kind=kind, orientation='horizontal')
    style = {'broll': {f"{item['media_type']}_placement": placement}}
    suggestions = suggest_broll(phrase_words(), [item], style, shape='horizontal')
    candidate = suggestions['phrases'][0]['candidates'][0]
    assert (candidate['kind'], candidate['placement'], candidate['box']) == (
        'insert', 'full_frame', None)
    write_document(tmp_path / 'project.json', {'shape': 'horizontal'})
    plan = plan_broll({'schema': 'yap-studio/broll-requests@1',
                       'broll': [candidate['request']]}, phrase_words(), [item], style,
                      project_dir=tmp_path, frames=90)
    assert plan.lane()['elements'][0]['kind'] == 'insert'


@pytest.mark.parametrize('width,height,accepted', [
    (1080, 1920, True), (1080, 1080, True), (608, 1080, True),
    (720, 720, False), (1920, 1080, True), (1280, 720, False),
])
def test_wide_stock_suitability_matches_render_fit(width, height, accepted):
    from yap_studio.stock.provider import StockFile, best_file, tier

    file = StockFile('1', 'https://videos.pexels.com/video-files/1/hd.mp4',
                     width, height, 30, 'hd')
    clip = SimpleNamespace(files=(file,), width=width, height=height)
    assert (best_file(clip, shape='horizontal') is not None) == accepted
    assert tier(clip, shape='horizontal') == ((0 if width > height else 1) if accepted else None)


@pytest.mark.parametrize('speed', [1, 2])
@pytest.mark.parametrize('timed', [True, False])
def test_wide_face_window_uses_presentation_times(tmp_path, monkeypatch, speed, timed):
    from yap_studio.faces import project as face_project

    # Trimming at 1.15 seconds starts at index 2, not average-fps index 4.
    times = (0., .1, 1.2, 1.4, 1.6, 1.8, 2.)
    frames = tuple((FaceBox(Box(400, 600 if i == 2 else 40, 100, 100), 1., 'detected'),)
                   for i in range(len(times)))
    faces = FacesData('measured', None,
        VideoIdentity('vfr.mov', 'a' * 64, len(times), Fraction(4), 1200, 900),
        {}, {}, frames, (), times=times if timed else None, end_time=2.2 if timed else None)
    monkeypatch.setattr(face_project, 'cut_faces', lambda *_: faces)
    source = SimpleNamespace(info=SimpleNamespace(width=1200, height=900, variable_frame_rate=True),
                             path=tmp_path / 'vfr.mov', image=False)
    item = ResolvedElement('broll-vfr', 'insert', 'broll', 0, 3, 'front',
                            {'media': {'in_s': 1.15, 'speed': speed}})
    result = wide_insert(item, source, tmp_path, (1920, 1080))
    if not timed:
        assert result.payload['media']['fit'] == 'contain'
        return
    x, y, w, h = result.payload['media']['crop']
    assert x <= 400 and y <= 540 and x + w >= 500 and y + h >= 700


@pytest.mark.parametrize('direction', ['in', 'out'])
@pytest.mark.parametrize('face_x,face_y,motion', [(900, 400, 'push'), (50, 10, 'still')])
def test_wide_face_safe_push_is_preserved(tmp_path, monkeypatch, direction, face_x, face_y,
                                         motion):
    from yap_studio.gallery import still

    face = SimpleNamespace(x=face_x, y=face_y, w=100, h=100)
    monkeypatch.setattr(still, 'default_landmark_detector',
                         lambda: SimpleNamespace(measure=lambda _: [[face]]))
    source = SimpleNamespace(info=SimpleNamespace(width=1920, height=1080),
                             path=tmp_path / 'photo.png', image=True, still=False)
    item = ResolvedElement('broll-push', 'insert', 'broll', 0, 6, 'front',
        {'media': {}, 'motion': 'push', 'push_scale': 1.08, 'push_direction': direction})
    result = wide_insert(item, source, tmp_path, (1920, 1080))
    assert result.payload['motion'] == motion


@pytest.mark.render
@pytest.mark.parametrize('case', ['grid', 'rotated-grid'])
def test_wide_prepared_heif_detects_upright_image_and_renders(tmp_path, monkeypatch, case):
    from yap_studio.gallery import still

    photo = heic_quadrants(tmp_path / 'photo.heic', width=1200, height=900)
    if case == 'rotated-grid':
        original = heic_quadrants(tmp_path / 'original.heic', width=900, height=1200)
        photo = tmp_path / 'photo.heif'
        subprocess.run(['swift', str(Path(__file__).parent / 'fixtures/heic_fixture.swift'),
                        str(original.with_suffix('.png')), str(photo), '6'],
                       check=True, capture_output=True)
    source = picture._source(photo, [], [], owner='broll-heif')
    item = ResolvedElement('broll-heif', 'insert', 'broll', 0, 3, 'front',
                           {'media': {}, 'motion': 'still'})
    with paths.operation():
        prepared = picture._prepare_still(source, tmp_path, [], [],
            source_sha256=receipt.sha256_file(photo), owner=item,
            ffmpeg_version=ffmpeg.tool_versions()['ffmpeg'])
        measured = []
        detector = still.default_landmark_detector()
        assert detector is not None

        def measure(images):
            assert images[0].suffix == '.png'
            info = ffmpeg.probe(images[0], count_frames=False)
            assert (info.width, info.height) == (prepared.info.width, prepared.info.height)
            measured.append(images[0])
            return detector.measure(images)

        monkeypatch.setattr(still, 'default_landmark_detector',
                             lambda: SimpleNamespace(measure=measure))
        result = wide_insert(item, prepared, tmp_path, (1920, 1080))
        assert len(measured) == 1
        graph = picture._Graph()
        branch = picture._insert(graph, prepared, result, size=(1920, 1080),
                                  delivery_size=(1920, 1080))
        target = tmp_path / 'wide.mp4'
        ffmpeg.run(['ffmpeg', '-v', 'error', '-filter_complex_threads', '1', *graph.inputs,
            '-filter_complex', ';'.join(graph.filters), '-map', f'[{branch}]',
            '-frames:v', '3', '-c:v', 'libx264', '-threads', '1', '-pix_fmt', 'yuv420p',
            str(target)], timeout_s=60)
        info = ffmpeg.probe(target)
        assert (info.width, info.height, info.video_frames) == (1920, 1080, 3)


@pytest.mark.parametrize('kind', ['image', 'video'])
def test_contact_sheet_reuses_each_shape_after_switch(tmp_path, monkeypatch, kind):
    from yap_studio.media import catalog, sheet

    source = (synth_broll.png(tmp_path / 'photo.png') if kind == 'image' else
              synth_broll.clip(tmp_path / 'clip.mov', duration=.3, audio=False))
    identifier = catalog.add_paths([source])['added'][0]
    vertical = sheet.make_sheet(identifier, asr='none')
    horizontal = sheet.make_sheet(identifier, asr='none', shape='horizontal')

    def unexpected(*args, **kwargs):
        pytest.fail('existing shape receipt should be reused')

    monkeypatch.setattr(sheet, '_render_still', unexpected)
    monkeypatch.setattr(sheet, '_render_clip', unexpected)
    for shape, expected in [('vertical', vertical), ('horizontal', horizontal)]:
        reused = sheet.make_sheet(identifier, asr='none', shape=shape)
        assert reused['noop'] and reused['sheet'] == expected['sheet']
        assert catalog.get_item(identifier)['sheet'] == expected['sheet']


def test_stock_cache_keeps_selected_renditions_separate(tmp_path):
    from yap_studio.stock.get import _obtain
    from yap_studio.stock.provider import StockFile

    files = tuple(StockFile(str(index), f'https://videos.pexels.com/video-files/1/{index}.mp4',
                            width, height, 30, 'hd')
                  for index, (width, height) in enumerate([(1920, 1080), (1080, 1920)]))
    clip = SimpleNamespace(files=files, duration_s=1, page_url='https://www.pexels.com/video/1/',
        author='Fixture', author_url='https://www.pexels.com/@fixture/',
        license='Pexels License', license_url='https://www.pexels.com/license/')
    calls = []
    provider = SimpleNamespace(name='pexels', label='Pexels', details=lambda _: clip)

    def fetch(url, target, **kwargs):
        file = next(file for file in files if file.url == url)
        calls.append(url)
        synth_broll.clip(target, size=f'{file.width}x{file.height}', duration=.3, audio=False)

    obtained = []
    for shape, file in [('horizontal', files[0]), ('vertical', files[1]),
                        ('horizontal', files[0]), ('vertical', files[1])]:
        item, target, cached, _ = _obtain(
            {'provider': 'pexels', 'clip_id': '1', 'file': {'file_id': file.file_id,
             'url': file.url, 'width': file.width, 'height': file.height}},
            {'window_s': .2, 'query': 'fixture'}, provider=provider, fetch=fetch, shape=shape)
        assert (item['width'], item['height']) == (file.width, file.height)
        obtained.append((target, receipt.sha256_file(target), cached))
    assert len(calls) == 2
    assert obtained[0][:2] == obtained[2][:2] and obtained[1][:2] == obtained[3][:2]
    assert obtained[0][0] != obtained[1][0]
    assert [entry[2] for entry in obtained] == [False, False, True, True]


def test_wide_stock_prefers_a_landscape_rendition():
    from yap_studio.stock.provider import StockFile, best_file

    files = tuple(StockFile(str(index), f'https://videos.pexels.com/video-files/1/{index}.mp4',
                            width, height, 30, 'hd')
                  for index, (width, height) in enumerate([(608, 1080), (1920, 1080)]))
    assert best_file(SimpleNamespace(files=files), shape='horizontal') == files[1]


def test_vfr_face_window_starts_at_first_frame_retained_by_trim(tmp_path, monkeypatch):
    from yap_studio.faces import project as face_project

    times = (1.2, 1.28, 1.6)
    frames = tuple((SimpleNamespace(box=Box(400, y, 100, 100)),) for y in (40, 600, 40))
    faces = SimpleNamespace(frames=frames, times=times, time_at=times.__getitem__)
    monkeypatch.setattr(face_project, 'cut_faces', lambda *_: faces)
    source = SimpleNamespace(info=SimpleNamespace(width=1200, height=900, variable_frame_rate=True),
                             path=tmp_path / 'vfr.mov', image=False)
    item = ResolvedElement('broll-vfr', 'insert', 'broll', 0, 3, 'front',
                            {'media': {'in_s': 1.15}})
    # trim removes the earlier frames; setpts makes 1.2 the zero, so 1.28 is visible.
    assert wide_insert(item, source, tmp_path, (1920, 1080)).payload['media']['fit'] == 'contain'
