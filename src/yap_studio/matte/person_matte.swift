import Foundation
import AVFoundation
import CoreMedia
import CoreVideo
import Vision
import Accelerate

// Stdout is exclusively packed gray frames. JSON records use a separate file.
enum MatteError: Error {
    case input(String)
    case transform(String)
    case vision(String)
}

func fail(_ message: String, code: Int32) -> Never {
    let line = message.components(separatedBy: .newlines).joined(separator: " ")
    FileHandle.standardError.write(Data((line + "\n").utf8))
    exit(code)
}

func decimal(_ value: Double) -> String {
    String(format: "%.6f", locale: Locale(identifier: "en_US_POSIX"), value)
}

func record(_ text: String, to file: FileHandle) throws {
    try file.write(contentsOf: Data((text + "\n").utf8))
}

func segment(_ frame: CVPixelBuffer, quality: String, revision: Int, index: Int) throws
    -> (bytes: [UInt8], maskWidth: Int, maskHeight: Int) {
    let request = VNGeneratePersonSegmentationRequest()
    request.qualityLevel = quality == "accurate" ? .accurate : .balanced
    request.outputPixelFormat = kCVPixelFormatType_OneComponent8
    request.revision = revision
    do {
        try VNImageRequestHandler(cvPixelBuffer: frame, options: [:]).perform([request])
    } catch {
        throw MatteError.vision("Vision failed at frame \(index): \(error.localizedDescription)")
    }
    guard let mask = request.results?.first?.pixelBuffer else {
        throw MatteError.vision("Vision returned no mask at frame \(index).")
    }
    let width = CVPixelBufferGetWidth(frame)
    let height = CVPixelBufferGetHeight(frame)
    let mw = CVPixelBufferGetWidth(mask)
    let mh = CVPixelBufferGetHeight(mask)
    guard CVPixelBufferLockBaseAddress(mask, .readOnly) == kCVReturnSuccess else {
        throw MatteError.vision("Cannot lock the Vision mask at frame \(index).")
    }
    defer { CVPixelBufferUnlockBaseAddress(mask, .readOnly) }
    guard let address = CVPixelBufferGetBaseAddress(mask) else {
        throw MatteError.vision("Vision mask has no pixels at frame \(index).")
    }
    var source = vImage_Buffer(data: address, height: vImagePixelCount(mh),
                               width: vImagePixelCount(mw), rowBytes: CVPixelBufferGetBytesPerRow(mask))
    var bytes = [UInt8](repeating: 0, count: width * height)
    let result = bytes.withUnsafeMutableBytes { pixels -> vImage_Error in
        var destination = vImage_Buffer(data: pixels.baseAddress!, height: vImagePixelCount(height),
                                         width: vImagePixelCount(width), rowBytes: width)
        return vImageScale_Planar8(&source, &destination, nil, vImage_Flags(kvImageHighQualityResampling))
    }
    guard result == kvImageNoError else {
        throw MatteError.vision("Cannot scale the Vision mask at frame \(index): \(result).")
    }
    return (bytes, mw, mh)
}

func makeMatte(video: String, records: String, quality: String) throws {
    guard FileManager.default.isReadableFile(atPath: video) else {
        throw MatteError.input("Cannot read video: \(video)")
    }
    let asset = AVURLAsset(url: URL(fileURLWithPath: video))
    guard let track = asset.tracks(withMediaType: .video).first else {
        throw MatteError.input("Video has no readable video track: \(video)")
    }
    guard track.preferredTransform.isIdentity else {
        throw MatteError.transform("Video must be an upright cut with an identity preferredTransform.")
    }
    let reader: AVAssetReader
    do {
        reader = try AVAssetReader(asset: asset)
    } catch {
        throw MatteError.input("Cannot open video reader: \(error.localizedDescription)")
    }
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
    ])
    output.alwaysCopiesSampleData = false
    guard reader.canAdd(output) else {
        throw MatteError.input("Cannot decode the video track as BGRA.")
    }
    reader.add(output)
    guard reader.startReading() else {
        throw MatteError.input(reader.error?.localizedDescription ?? "Cannot start video reader.")
    }
    guard FileManager.default.createFile(atPath: records, contents: nil) else {
        throw MatteError.input("Cannot create records file: \(records)")
    }
    let file = try FileHandle(forWritingTo: URL(fileURLWithPath: records))
    defer { try? file.close() }
    let revision = VNGeneratePersonSegmentationRequest.supportedRevisions.max()
        ?? VNGeneratePersonSegmentationRequest.defaultRevision
    var index = 0
    var width = 0
    var height = 0
    var previous: Double?
    var positiveStep: Double?
    var sampleDuration = Double.nan
    while let sample = output.copyNextSampleBuffer() {
        try autoreleasepool {
            guard let frame = CMSampleBufferGetImageBuffer(sample) else {
                throw MatteError.input("Missing decoded pixels at frame \(index).")
            }
            let pts = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sample))
            guard pts.isFinite, previous == nil || pts >= previous! else {
                throw MatteError.input("Invalid presentation time at frame \(index).")
            }
            if let last = previous, pts > last { positiveStep = pts - last }
            previous = pts
            sampleDuration = CMTimeGetSeconds(CMSampleBufferGetDuration(sample))
            let w = CVPixelBufferGetWidth(frame)
            let h = CVPixelBufferGetHeight(frame)
            guard w > 0, h > 0, index == 0 || (w == width && h == height) else {
                throw MatteError.input("Invalid or changing frame dimensions at frame \(index).")
            }
            let mask = try segment(frame, quality: quality, revision: revision, index: index)
            if index == 0 {
                width = w
                height = h
                let header: [String: Any] = [
                    "type": "header", "width": w, "height": h, "quality": quality,
                    "revision": revision, "mask_width": mask.maskWidth, "mask_height": mask.maskHeight
                ]
                let data = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
                try record(String(decoding: data, as: UTF8.self), to: file)
            }
            let foreground = mask.bytes.reduce(0) { $0 + ($1 >= 128 ? 1 : 0) }
            let coverage = Double(foreground) / Double(mask.bytes.count)
            try FileHandle.standardOutput.write(contentsOf: Data(mask.bytes))
            try record("{\"coverage\":\(decimal(coverage)),\"frame\":\(index),"
                       + "\"pts\":\(decimal(pts)),\"type\":\"frame\"}", to: file)
            index += 1
        }
    }
    guard reader.status == .completed, index > 0, let last = previous else {
        throw MatteError.input(reader.error?.localizedDescription ?? "Video has no decoded frames.")
    }
    let nominal = 1 / Double(track.nominalFrameRate)
    let duration: Double
    if sampleDuration.isFinite && sampleDuration > 0 {
        duration = sampleDuration
    } else if let step = positiveStep, step.isFinite && step > 0 {
        duration = step
    } else if nominal.isFinite && nominal > 0 {
        duration = nominal
    } else {
        duration = 0
    }
    guard (last + duration).isFinite else {
        throw MatteError.input("Invalid video end presentation time.")
    }
    try record("{\"end_pts\":\(decimal(last + duration)),\"frames\":\(index),\"type\":\"end\"}",
               to: file)
}

let usage = "Usage: yap-person-matte [--quality accurate|balanced] --records RECORDS.jsonl VIDEO"
let args = Array(CommandLine.arguments.dropFirst())
var quality = "accurate"
var records: String?
var video: String?
var cursor = 0
while cursor < args.count {
    let argument = args[cursor]
    if argument == "--quality" || argument == "--records" {
        guard cursor + 1 < args.count else { fail(usage, code: 2) }
        if argument == "--quality" { quality = args[cursor + 1] }
        else { records = args[cursor + 1] }
        cursor += 2
    } else {
        guard !argument.hasPrefix("--"), video == nil else { fail(usage, code: 2) }
        video = argument
        cursor += 1
    }
}
guard let input = video, !input.isEmpty, let destination = records, !destination.isEmpty,
      quality == "accurate" || quality == "balanced" else { fail(usage, code: 2) }
do {
    try makeMatte(video: input, records: destination, quality: quality)
} catch MatteError.input(let message) {
    fail(message, code: 2)
} catch MatteError.transform(let message) {
    fail(message, code: 3)
} catch MatteError.vision(let message) {
    fail(message, code: 4)
} catch {
    fail("Person segmentation failed: \(error.localizedDescription)", code: 4)
}
