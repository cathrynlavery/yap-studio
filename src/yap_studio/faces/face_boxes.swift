import Foundation
import AVFoundation
import CoreMedia
import CoreVideo
import Vision
import ImageIO

// The helper emits only protocol records on stdout. Diagnostics go to stderr.
enum HelperError: Error {
    case unreadable(String)
    case detection(String)
}

func fail(_ message: String, code: Int32) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(code)
}

func emit(_ record: [String: Any]) throws {
    let data = try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys])
    print(String(decoding: data, as: UTF8.self))
}

func number(_ value: Double) -> String {
    return String(format: "%.3f", locale: Locale(identifier: "en_US_POSIX"), value)
}

func seconds(_ value: Double) -> String {
    return String(format: "%.6f", locale: Locale(identifier: "en_US_POSIX"), value)
}

func orientation(_ transform: CGAffineTransform) -> CGImagePropertyOrientation {
    let xScale = hypot(transform.a, transform.b)
    let yScale = hypot(transform.c, transform.d)
    if xScale.isFinite && xScale > 0 && yScale.isFinite && yScale > 0 {
        let a = Int((transform.a / xScale).rounded())
        let b = Int((transform.b / xScale).rounded())
        let c = Int((transform.c / yScale).rounded())
        let d = Int((transform.d / yScale).rounded())
        // Increasing y lands at (c, d), increasing x at (a, b) in display space.
        // Each case names the display edges containing stored row 0 and column 0.
        switch (a, b, c, d) {
        case (1, 0, 0, 1): return .up                 // top, left
        case (-1, 0, 0, 1): return .upMirrored        // top, right
        case (-1, 0, 0, -1): return .down             // bottom, right
        case (1, 0, 0, -1): return .downMirrored      // bottom, left
        case (0, 1, 1, 0): return .leftMirrored       // left, top
        case (0, 1, -1, 0): return .right             // right, top
        case (0, -1, -1, 0): return .rightMirrored    // right, bottom
        case (0, -1, 1, 0): return .left              // left, bottom
        default: break
        }
    }
    FileHandle.standardError.write(Data(
        "Warning: unsupported video transform; using up orientation.\n".utf8
    ))
    return .up
}

func orientationName(_ value: CGImagePropertyOrientation) -> String {
    switch value {
    case .up: return "up"
    case .upMirrored: return "upMirrored"
    case .down: return "down"
    case .downMirrored: return "downMirrored"
    case .left: return "left"
    case .leftMirrored: return "leftMirrored"
    case .right: return "right"
    case .rightMirrored: return "rightMirrored"
    @unknown default: return "up"
    }
}

func finalFrameDuration(sample: Double, lastPositiveStep: Double?, nominal: Double) -> Double {
    if sample.isFinite && sample > 0 {
        return sample
    }
    if let step = lastPositiveStep, step.isFinite && step > 0 {
        return step
    }
    if nominal.isFinite && nominal > 0 {
        return nominal
    }
    FileHandle.standardError.write(Data(
        "Warning: no valid final frame duration; using zero duration.\n".utf8
    ))
    return 0
}

func emitSkippedFrame(index: Int, pts: CMTime) throws {
    let time = CMTimeGetSeconds(pts)
    guard time.isFinite else {
        throw HelperError.unreadable("Video has an invalid presentation time at frame \(index).")
    }
    print("{\"type\":\"frame\",\"frame\":\(index),\"pts\":\(seconds(time)),\"faces\":null}")
}

func detectFrame(
    _ buffer: CVPixelBuffer, index: Int, pts: CMTime,
    width: Int, height: Int, orientation: CGImagePropertyOrientation, revision: Int
) throws {
    let request = VNDetectFaceRectanglesRequest()
    request.revision = revision
    let handler = VNImageRequestHandler(cvPixelBuffer: buffer, orientation: orientation, options: [:])
    do {
        try handler.perform([request])
    } catch {
        throw HelperError.detection("Vision failed at frame \(index): \(error.localizedDescription)")
    }
    let w = Double(width)
    let h = Double(height)
    let observations = (request.results ?? []).sorted {
        if $0.boundingBox.maxY != $1.boundingBox.maxY {
            return $0.boundingBox.maxY > $1.boundingBox.maxY
        }
        return $0.boundingBox.minX < $1.boundingBox.minX
    }
    let faces = observations.map { observation -> String in
        let box = observation.boundingBox
        let x = max(0, min(w, Double(box.minX) * w))
        let y = max(0, min(h, (1 - Double(box.maxY)) * h))
        let right = max(x, min(w, Double(box.maxX) * w))
        let bottom = max(y, min(h, (1 - Double(box.minY)) * h))
        let confidence = max(0, min(1, Double(observation.confidence)))
        return "{\"x\":\(number(x)),\"y\":\(number(y)),\"w\":\(number(right - x)),"
            + "\"h\":\(number(bottom - y)),\"confidence\":\(number(confidence))}"
    }
    let time = CMTimeGetSeconds(pts)
    guard time.isFinite else {
        throw HelperError.unreadable("Video has an invalid presentation time at frame \(index).")
    }
    print("{\"type\":\"frame\",\"frame\":\(index),\"pts\":\(seconds(time)),"
          + "\"faces\":[\(faces.joined(separator: ","))]}")
}

func run(_ video: String, every: Int) throws {
    guard FileManager.default.isReadableFile(atPath: video) else {
        throw HelperError.unreadable("Cannot read video: \(video)")
    }
    let asset = AVURLAsset(url: URL(fileURLWithPath: video))
    guard let track = asset.tracks(withMediaType: .video).first else {
        throw HelperError.unreadable("Video has no readable video track: \(video)")
    }
    let transform = track.preferredTransform
    let displaySize = track.naturalSize.applying(transform)
    let width = Int(abs(displaySize.width).rounded())
    let height = Int(abs(displaySize.height).rounded())
    guard width > 0, height > 0 else {
        throw HelperError.unreadable("Video has invalid display dimensions.")
    }
    let imageOrientation = orientation(transform)
    let revision = VNDetectFaceRectanglesRequest.supportedRevisions.max()
        ?? VNDetectFaceRectanglesRequest.defaultRevision
    let reader: AVAssetReader
    do {
        reader = try AVAssetReader(asset: asset)
    } catch {
        throw HelperError.unreadable("Cannot read video: \(error.localizedDescription)")
    }
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
    ])
    output.alwaysCopiesSampleData = false
    guard reader.canAdd(output) else {
        throw HelperError.unreadable("Cannot decode the first video track.")
    }
    reader.add(output)
    guard reader.startReading() else {
        throw HelperError.unreadable(reader.error?.localizedDescription ?? "Cannot start video reader.")
    }
    try emit(["type": "header", "width": width, "height": height, "revision": revision])
    var index = 0
    var lastPTS: Double?
    var lastPositiveStep: Double?
    var lastDuration = Double.nan
    let nominalDuration = 1 / Double(track.nominalFrameRate)
    // Hold one skipped frame so the final frame can be sampled without reordering.
    var lastUnsampled: (index: Int, pts: CMTime, buffer: CVPixelBuffer)?
    while let sample = output.copyNextSampleBuffer() {
        try autoreleasepool {
            guard let buffer = CMSampleBufferGetImageBuffer(sample) else {
                throw HelperError.unreadable("Video has no image buffer at frame \(index).")
            }
            let pts = CMSampleBufferGetPresentationTimeStamp(sample)
            let time = CMTimeGetSeconds(pts)
            guard time.isFinite, lastPTS == nil || time >= lastPTS! else {
                throw HelperError.unreadable("Video has an invalid presentation time at frame \(index).")
            }
            if let previous = lastPTS {
                let step = time - previous
                if step > 0 {
                    lastPositiveStep = step
                }
            }
            lastPTS = time
            lastDuration = CMTimeGetSeconds(CMSampleBufferGetDuration(sample))
            if let held = lastUnsampled {
                try emitSkippedFrame(index: held.index, pts: held.pts)
                lastUnsampled = nil
            }
            if index % every == 0 {
                try detectFrame(buffer, index: index, pts: pts, width: width, height: height,
                                orientation: imageOrientation, revision: revision)
            } else {
                lastUnsampled = (index, pts, buffer)
            }
            index += 1
        }
    }
    guard reader.status == .completed, index > 0 else {
        throw HelperError.unreadable(reader.error?.localizedDescription ?? "Video has no decoded frames.")
    }
    if let last = lastUnsampled {
        try detectFrame(last.buffer, index: last.index, pts: last.pts, width: width, height: height,
                        orientation: imageOrientation, revision: revision)
        lastUnsampled = nil
    }
    let duration = finalFrameDuration(sample: lastDuration, lastPositiveStep: lastPositiveStep,
                                      nominal: nominalDuration)
    guard let last = lastPTS, (last + duration).isFinite else {
        throw HelperError.unreadable("Video has an invalid end presentation time.")
    }
    print("{\"type\":\"end\",\"frames\":\(index),\"end_pts\":\(seconds(last + duration))}")
    fflush(stdout)
}

let arguments = CommandLine.arguments
if arguments.dropFirst().first == "--orientation" {
    let values = arguments.dropFirst(2).compactMap { Double($0) }
    guard arguments.count == 6, values.count == 4, values.allSatisfy({ $0.isFinite }) else {
        fail("Usage: yap-face-boxes --orientation A B C D (finite numbers required)", code: 2)
    }
    let transform = CGAffineTransform(a: CGFloat(values[0]), b: CGFloat(values[1]),
                                      c: CGFloat(values[2]), d: CGFloat(values[3]), tx: 0, ty: 0)
    do {
        try emit(["orientation": orientationName(orientation(transform))])
    } catch {
        fail("Cannot emit orientation: \(error.localizedDescription)", code: 4)
    }
    exit(0)
}
if arguments.dropFirst().first == "--final-duration" {
    let usage = "Usage: yap-face-boxes --final-duration SAMPLE STEP NOMINAL (STEP may be none)"
    guard arguments.count == 5,
          let sample = Double(arguments[2]), let nominal = Double(arguments[4]) else {
        fail(usage, code: 2)
    }
    let step: Double?
    if arguments[3] == "none" {
        step = nil
    } else {
        guard let value = Double(arguments[3]) else {
            fail(usage, code: 2)
        }
        step = value
    }
    let duration = finalFrameDuration(sample: sample, lastPositiveStep: step, nominal: nominal)
    print("{\"duration\":\(seconds(duration))}")
    exit(0)
}
guard arguments.count == 4, arguments[1] == "--every",
      let every = Int(arguments[2]), every >= 1, !arguments[3].isEmpty else {
    fail("Usage: yap-face-boxes --every N VIDEO (N must be >= 1)", code: 2)
}
do {
    try run(arguments[3], every: every)
} catch HelperError.unreadable(let message) {
    fail(message, code: 3)
} catch HelperError.detection(let message) {
    fail(message, code: 4)
} catch {
    fail("Face detection failed: \(error.localizedDescription)", code: 4)
}
