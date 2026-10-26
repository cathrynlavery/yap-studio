# The style page

The style page is where you choose how your videos look: fonts, captions, colors, the
color grade, the opening title, cards, B-roll, camera moves, moments and sound. You pick
once, and every new video gets the style you save until you change it.

Open it from your workspace:

```console
yap-studio style pick
```

Add `--phone` to open it on your phone; [Review](review.md) explains how phone links
work. [Styles](styles.md#pick-your-look) covers `--detach` and `--stop`.

## How the page works

The page opens on looks under your video. Your saved styles come first, newest first,
with the names you gave them. The presets follow: Editorial Serif, Clean Minimal, Warm
Editorial, Cobalt, Mono, Headline, Electric, Candy, Marker, One Word, Bold, Editorial
and Neutral. A look sets every choice, from fonts to sound. Tap a look to see it on
your take. If it is right, press Save style, answer Text at the top, and save it.

Customize opens four screens: Font, Captions, Colors and Text at the top.
Back and Next move through them, then on to More choices and the
optional screens. Back from Font returns to looks. Back to looks returns to the grid and
keeps your changes. Reset changes puts the look back as it was. The step bar shows
Font, Captions, Colors, Text at the top and More choices.

More choices lists Grade, Opening, Cards, B-roll, Moves, Moments and Sound. Each row
shows a still preview, the current choice and Change. Nothing opens until you tap a
row. The look already set all of these, so open only the ones you want to change. A
screen stays hidden while it has nothing to offer, and joins the list once it does.
Each optional screen starts with All choices, which takes you back to the list.

Each screen puts its main choices first. More starts closed and has a short hint of
what is inside. Its rows start closed too and show their current values. The page
remembers which folds you opened until you close it. Opening a screen or a fold changes
nothing in your style. Only a choice does.

Save style is always in the bottom bar. Back and Next keep your choices, and Next
hides on the last screen you can reach.

On a computer the preview stays beside the controls. On a phone it stays small, and a
tap enlarges it. Nothing scrolls sideways. Caption, title and card tiles play the
engine's own render over your still, with a poster while it loads. On a computer, hover
a tile to play it. On a phone, tiles play while they are on screen. B-roll, Moves,
Moments and Sound tiles draw over the still without a render, and previews have no
sound. Motion previews respect your reduced motion setting.

3D camera draws its preview over your still. The word behind your head appears
when the person cutout is ready.

### Font

Pick the caption font from the grid. Each name is drawn in its own font. When
captions, titles and cards share a font, one tap changes all three. A style with
separate title and card fonts keeps them until you change them. Open More, then Titles
and cards font, to keep Same as captions, use the Instrument Serif and Geist pairing,
or pick another font. Letter width has its own fold, and a custom width stays exact.
Fonts you add with `yap-studio brand fonts add FILE` join the grid.

### Captions

Plain, Box, Band, One word, Rise, Word by word and 3D camera are the main treatments.
Word by word shows each word when it is said and keeps earlier words on screen.
The live preview builds the phrase with fixed lines and a full size background.
More holds words on screen, all 24 highlights, position, size, color and shadow,
background color, phrase motion and exact values. Italicize emphasized words is
with the highlights. Keep captions off my face lets the engine move captions
around the faces it measures. If there is no clear position, captions stops with
the times. Turning it off saves `captions.placement: "fixed"` and accepts captions
over your face. The final reports those overlaps as warnings. Cards and overlays
still have to clear your face.

3D camera puts words at three depths around you and moves the picture with a virtual
camera. It uses your caption font, color, shadow and phrase limits, plus your accent
color and title font for the big word behind you. It needs a steady vertical
take with you in every shot.

### Colors

Pick one of ten palettes or Your colors. More holds the four hex fields (Captions,
Highlight, Cards and Card text) and Pull colors from a website. The reader only reads
public sites, and a black and white site keeps your highlight color. A color warning
stays under the tiles when a choice may be hard to read.

### Grade

None keeps the picture as shot. Film look adds warmer, softer contrast. Eyedropper
matches a color you pick on the preview or type as hex. More holds Light or Full
strength and the before and after switch. The tiles use the graded still, and captions
and cards keep their own colors. The style file fields are in
[Styles](styles.md#picture-colour).

### Text at the top

Answer one question: when would you like text at the top of your video?
Choose None, Just the opening, or The whole video. Each previews on your own still.
Just the opening uses your title look for the first few seconds. The whole video
uses a short label like "Security rant". Nothing is selected for you, and nothing
appears at the top until you answer. Save style opens this question if needed.
For The whole video, Label color offers White, Black or Accent for the box.
Its preview uses a short line from your take when words are available.

For either kind, the agent writes three ideas for each video, shows them in the
Story cut note and on the review page, and waits for your pick or your own text.
It never chooses for you. Older saved styles keep rendering as before, and the
agent asks this question once before your next cut.

Your answer saves `hook.answered: true`. `hook.style` is `none` for None,
your title look for Just the opening, or `text_hook` for The whole video.
Older styles resolve with `hook.answered: false` and keep their existing look.

### Opening look

The look's own title comes first, then Calm card, Headline slam, Shimmer, Typewriter,
Video in letters. Video in letters fills a big word with your footage.
More holds the other title choices, and a note says how long the title shows.
You choose the text for each video later.

### Cards

Choose Paper, Minimal, Dark or Bright. Open a kind group to change its switches, or use
All for the whole group. Card groups load their rows and previews only while open. Tap
a kind's picture to preview it. Stat shows sample values. Numbers includes Big number
behind you, which counts up behind your head. It starts off, and its preview uses the
cached person cutout once that is ready. New card kinds start off. Dark saves dark
paper with light ink; Paper and Minimal bring back a light pair.
Words includes Choices, a list of your spoken options that glows in your accent color.
It starts off.
Some kinds come from HyperFrames registry items that setup installs. Like every new
kind, they start off. They preview in your font and colours, with a dark edge so
they stay readable on a light wall.
Flow look sits below the kind groups. Boxes draws the steps in a card. Tags stacks
bare labels with an arrow, and each tag pops in on its word.
Specs list rows offers Together, Rows on cue, or Rows on cue, glowing as card previews.
Card motion sits below it. None keeps each card's usual entrance. Motion streak flies
supported cards in and out with a fading trail. Number cards and full-screen cards enter as usual.

### B-roll

Choose Full screen, Mixed, In a box or Split. Split puts the shot on top and you below.
More holds placement for each kind of shot, box position, box motion, slow move on
photos and Stock clips. Match color to me shifts each shot toward your take's color. It
starts on and never touches your face. Ask first shows you a pick sheet before any
stock clip is used. The Video in letters switch starts off. Turn it on to allow a big
word filled with your footage on a key line.
Tall, behind your head puts long pages behind you.
Scroll moves a long page up in eased steps and leaves shorter shots still.
Framing behind the tall box keeps your room and framing, shrinks you in front of the
page, or chooses automatically from the space behind your head.

### Moves

Choose No zooms, Punches or Punches + slow push. A punch moves in fast on a key word,
holds, then moves out at the end of the phrase. Slow push adds one gentle push on a
longer line. More shows Zoom strength, 1.5x or 2x, only while zooms are on, and No
zooms keeps the strength you chose. Transitions is always in More. Its Whip, Zoom and
Light leak switches choose which cuts between shots the agent may use. All three start
on. Turn them all off for straight cuts. Each small preview cuts from your still to a
sample shot.

Jump cuts in More offers Hard cut, Punch, Smooth (punch on big moves) and Step in.
It starts on Hard cut. Step in changes the framing on cuts in small steps, then returns wide.

More also holds Push in on the first line. It starts off, and its preview pushes in
and cuts back wide. The push ends at the first sentence end or real cut between
0.8 and 4 seconds after the first word. If the line runs longer, it uses the first
clause end in that window. Longer lines push more slowly and reach 1.3x at their
end. If no boundary fits, the push is skipped with a warning. It never cuts wide
in the middle of a clause.

### Moments

Try Color pop, Background blur, Spotlight, Sticker freeze, Outline glow or Clone flash. A tile
previews the effect without turning it on. Use in my videos changes only that moment's
switch. They all start off, and the agent uses at most two moments in a video. On a Mac
the page caches a cutout of you from the preview still, and moments use it to separate
you from the room. Until it is ready, the effect covers the whole picture.
Clone flash is unavailable on green or blue screen takes, and the engine refuses it there too.
The page checks your cutout choice, the newest take or your saved video type.

### Sound

Choose No music, Quiet, Medium or Loud. More holds exact volume, sound effects, beat
snap, bleep and voice clean-up. Sound effects has three choices: Made by Yap Studio, which needs
no account, ElevenLabs, which brings back the look's ElevenLabs kit, and Off. With
ElevenLabs picked, a switch uses Made by Yap Studio until your ElevenLabs kit is ready.
That switch starts on. More sounds offers Typing, Message ping and Camera shutter,
all off. Applause, laugh, cheer and record scratch play only when you ask for them.
Music dips while you talk. Changing the exact volume updates its value and meter
without rebuilding the previews.
Bleep swear words, also under More, starts off. When it is on, a tone covers each swear
word, and captions, titles and cards show it as f***.

## Saving

Save style opens a name sheet with "My style" filled in, or the saved name if you
started from your library. Answer Text at the top first if you have not yet.
Confirm Save style to finish. Saving puts the style in your
library and makes it the default for new videos in one step. Each save is a new
version, so videos you already made keep their look. In a workspace, a readable copy
goes in `style/my-style.json`. The saved screen confirms that your next videos use it,
and Update my style takes you back to the page.

Each new video gets the style it names, else your default, else Clean Minimal. Run
`yap-studio style add FILE` to add a style file someone sent you.

The still behind the previews comes from the newest take in `inbox/`. Yap Studio picks
a frame with open eyes and room for captions, and the first complete sentence of at
least five words that does not start with a filler or a greeting. With an empty inbox
the page uses a drawn sample frame. The page and the render use the same fonts, and
none of them load from the web.

## Use a style without the page

Caption one vertical clip in a saved style:

```console
yap-studio captions CLIP --style STYLE --out OUT.mp4
```

Add `--title "Opening words|Key words"` for the opening title and `--srt` for a
subtitle file. In a project with a finished cut, `yap-studio captions DIR` renders the
captions and `yap-studio graphics DIR` renders the timeline titles, both in the
project's saved style. After you change the style, `yap-studio status DIR` shows what
needs rendering again.

## Start from videos you already posted

Run `yap-studio taste learn VIDEO... --out DIR` on finished vertical videos. It writes
a proposed style and proposed taste rules. `DIR/measurements.json` shows each measured
value and the frames behind it. Use `--base preset:electric` or `--base FILE` to choose
the style for anything it cannot measure; without `--base` it uses the schema
defaults. Finished videos suggest values, but they can't tell which choices you want to
repeat, so read the proposals before you keep them. Fonts and words per caption are not
measured yet.

Add `--save` to keep the result in one step. The proposed look goes into your library
as "Learned from your videos" (id `learned`) and the proposed rules wait in your
pending taste proposals. It never becomes your default on its own; you pick it in the
gallery or you don't. Approve rules one at a time with `yap-studio taste approve ID`.
Learning again replaces the learned style unless you made it your default, and it
never touches styles you saved yourself. When it can't replace it, the new one gets the
next free id: `learned-2`, `learned-3` and so on. A `.source.json` record next to each
learned style says how many finished videos it came from.

Without `--save`, add the look with `yap-studio style add DIR/proposed-style.json` and
the rules with `yap-studio taste add-proposal DIR/taste-proposals.json`. Nothing applies
until you add the style or approve a rule.

## What the choices write

The page writes the same style file you can edit by hand
([Styles](styles.md#style-files)).

3D camera sets `captions.camera_3d` to `true`. Picking another caption treatment
sets it to `false`.

- Shadow Soft is x 0, y 3, blur 8, opacity 0.35. Strong is x 0, y 6, blur 3, opacity
  0.8. Off sets the opacity to 0, and On brings a custom shadow back exactly.
- Top, Middle and Bottom set `top_px` to 700, 1180 and 1500. That is the preferred
  spot. Keep captions off my face lets the engine move captions off a face. The render
  measures faces through each phrase, and the preview uses the faces in the still.
  With it on, a phrase with no clear position stops captions before rendering.
- A background below 50% strength keeps your caption color. Stronger backgrounds use
  the engine's contrast color. A background at zero strength lays out like None.
- `captions.width_scale` runs from 0.6 to 1. `captions.highlight_style` can be `pill`
  (words light up inside a soft pill), `gradient` (each word fills as it is spoken) or
  `editorial` (the stressed word stays bold for the whole phrase and takes the
  highlight color once spoken).

## If you get stuck

- One word at a time is a main choice in Captions. The other Words on screen choices
  are in Captions under More.
- For plain captions, pick No highlight in Captions under More. Italics turn off too, and you can turn them
  back on.
- If website colors look wrong, the site may block the reader, and a black and white
  site has no highlight color to find. Type the four hex codes instead. The reader only
  reads public sites.
- Fonts can take a moment to load on a phone. The page shows the same fonts as your
  finished video.
- A dark highlight used as a colored word can disappear on video. Pick an accent that
  stays readable.
- A look does not need adjusting. Save it as it is, or open Customize to change it.
- Changing your style changes only the videos you make after you save.

## Credits

Bold, Editorial and Neutral come from HyperFrames theme packs (Apache-2.0, commit
`6afcb48395632b85663b9b9486e1b47fd175f8bc`); their provenance is in
`src/yap_studio/style/PROVENANCE.json`. Bold uses the bundled Anton SC in place of
Impact and keeps Inter for card and body text. Editorial uses the bundled Gelasio in
place of Georgia and Iowan Old Style. Neutral keeps the bundled Inter. Theme colors map
to caption, title and card colors. Theme settings with no matching style field are
recorded and left out of the engine settings.
