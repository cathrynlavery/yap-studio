import Foundation
import CoreVideo
import CoreMedia
import VideoToolbox

struct Failure: Error, CustomStringConvertible {
    let description: String
    init(_ message: String) { description = message }
}

func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw Failure(message) }
}

@available(macOS 15.4, *)
func makeBuffer(width: Int, height: Int, attributes: [String: Any]) throws -> CVPixelBuffer {
    var attributes = attributes
    attributes[kCVPixelBufferIOSurfacePropertiesKey as String] = [:] as [String: Any]
    attributes[kCVPixelBufferPixelFormatTypeKey as String] = kCVPixelFormatType_64RGBAHalf
    var buffer: CVPixelBuffer?
    let code = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                  kCVPixelFormatType_64RGBAHalf, attributes as CFDictionary, &buffer)
    guard code == kCVReturnSuccess, let buffer else {
        throw Failure("Cannot create half-float pixel buffer: \(code).")
    }
    return buffer
}

func fill(_ buffer: CVPixelBuffer, raw: Data, width: Int, height: Int) throws {
    try require(CVPixelBufferLockBaseAddress(buffer, []) == kCVReturnSuccess,
                "Cannot lock source pixel buffer.")
    let stride = CVPixelBufferGetBytesPerRow(buffer) / 2
    guard let address = CVPixelBufferGetBaseAddress(buffer) else {
        CVPixelBufferUnlockBaseAddress(buffer, [])
        throw Failure("Missing source pixel storage.")
    }
    let destination = address.assumingMemoryBound(to: UInt16.self)
    raw.withUnsafeBytes { bytes in
        let input = bytes.bindMemory(to: UInt16.self)
        for y in 0..<height {
            for x in 0..<width {
                for c in 0..<3 {
                    let value = UInt16(littleEndian: input[(y * width + x) * 3 + c])
                    destination[y * stride + x * 4 + c] = Float16(Float(value) / 65535).bitPattern
                }
                destination[y * stride + x * 4 + 3] = Float16(1).bitPattern
            }
        }
    }
    try require(CVPixelBufferUnlockBaseAddress(buffer, []) == kCVReturnSuccess,
                "Cannot unlock source pixel buffer.")
}

func read(_ buffer: CVPixelBuffer, width: Int, height: Int) throws -> Data {
    try require(CVPixelBufferLockBaseAddress(buffer, .readOnly) == kCVReturnSuccess,
                "Cannot lock destination pixel buffer.")
    let stride = CVPixelBufferGetBytesPerRow(buffer) / 2
    guard let address = CVPixelBufferGetBaseAddress(buffer) else {
        CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
        throw Failure("Missing destination pixel storage.")
    }
    let input = address.assumingMemoryBound(to: UInt16.self)
    var raw = Data(count: width * height * 6)
    raw.withUnsafeMutableBytes { bytes in
        let output = bytes.bindMemory(to: UInt16.self)
        for y in 0..<height {
            for x in 0..<width {
                for c in 0..<3 {
                    let value = Float(Float16(bitPattern: input[y * stride + x * 4 + c]))
                    let clamped = value.isFinite ? min(1, max(0, value)) : 0
                    output[(y * width + x) * 3 + c] = UInt16((clamped * 65535).rounded()).littleEndian
                }
            }
        }
    }
    try require(CVPixelBufferUnlockBaseAddress(buffer, .readOnly) == kCVReturnSuccess,
                "Cannot unlock destination pixel buffer.")
    return raw
}

@available(macOS 15.4, *)
func run() throws {
    let arguments = Array(CommandLine.arguments.dropFirst())
    if arguments == ["--check"] {
        print("{\"supported\":\(VTFrameRateConversionConfiguration.isSupported),\"revision\":1}")
        return
    }
    if !VTFrameRateConversionConfiguration.isSupported {
        FileHandle.standardError.write(Data("Frame rate conversion processor is not supported on this Mac.\n".utf8))
        exit(3)
    }
    try require(arguments.count == 6, "Use --width W --height H --phases p1,p2,...")
    var options: [String: String] = [:]
    for index in stride(from: 0, to: arguments.count, by: 2) {
        let key = arguments[index]
        try require(["--width", "--height", "--phases"].contains(key) && options[key] == nil,
                    "Unknown or repeated option.")
        options[key] = arguments[index + 1]
    }
    guard let width = Int(options["--width"] ?? ""),
          let height = Int(options["--height"] ?? ""),
          width > 0, width <= 8192, height > 0, height <= 4320 else {
        throw Failure("Invalid frame dimensions.")
    }
    let values = (options["--phases"] ?? "").split(separator: ",", omittingEmptySubsequences: false)
    let phases = try values.map { value -> Float in
        guard let phase = Float(value), phase.isFinite, phase > 0, phase < 1 else {
            throw Failure("Phases must be finite and strictly between zero and one.")
        }
        return phase
    }
    try require(!phases.isEmpty && zip(phases, phases.dropFirst()).allSatisfy { $0 < $1 },
                "Phases must be strictly increasing.")
    guard let configuration = VTFrameRateConversionConfiguration(
        frameWidth: width, frameHeight: height, usePrecomputedFlow: false,
        qualityPrioritization: .quality, revision: .revision1) else {
        throw Failure("Frame rate conversion configuration refused these dimensions.")
    }
    let size = width * height * 6
    let raw = FileHandle.standardInput.readData(ofLength: size * 2 + 1)
    try require(raw.count == size * 2, "Input must contain exactly two rgb48le frames.")
    let source = try makeBuffer(width: width, height: height,
                                attributes: configuration.sourcePixelBufferAttributes)
    let next = try makeBuffer(width: width, height: height,
                              attributes: configuration.sourcePixelBufferAttributes)
    try fill(source, raw: raw.subdata(in: 0..<size), width: width, height: height)
    try fill(next, raw: raw.subdata(in: size..<size * 2), width: width, height: height)
    let duration = Double(phases.count + 1) / 30
    guard let sourceFrame = VTFrameProcessorFrame(buffer: source, presentationTimeStamp: .zero),
          let nextFrame = VTFrameProcessorFrame(buffer: next, presentationTimeStamp:
                                                CMTime(seconds: duration, preferredTimescale: 600000)) else {
        throw Failure("Cannot wrap source pixel buffers.")
    }
    let destinations = try phases.map { phase -> VTFrameProcessorFrame in
        let buffer = try makeBuffer(width: width, height: height,
                                     attributes: configuration.destinationPixelBufferAttributes)
        guard let frame = VTFrameProcessorFrame(buffer: buffer, presentationTimeStamp:
            CMTime(seconds: Double(phase) * duration, preferredTimescale: 600000)) else {
            throw Failure("Cannot wrap destination pixel buffer.")
        }
        return frame
    }
    guard let parameters = VTFrameRateConversionParameters(
        sourceFrame: sourceFrame, nextFrame: nextFrame, opticalFlow: nil,
        interpolationPhase: phases, submissionMode: .random, destinationFrames: destinations) else {
        throw Failure("Frame rate conversion parameters were refused.")
    }
    let processor = VTFrameProcessor()
    try processor.startSession(configuration: configuration)
    defer { processor.endSession() }
    let semaphore = DispatchSemaphore(value: 0)
    var processingError: Error?
    processor.process(parameters: parameters) { _, error in
        processingError = error
        semaphore.signal()
    }
    semaphore.wait()
    if let processingError { throw processingError }
    for frame in destinations {
        try FileHandle.standardOutput.write(contentsOf: read(frame.buffer, width: width, height: height))
    }
}

do {
    if #available(macOS 15.4, *) {
        try run()
    } else {
        throw Failure("Update macOS to 15.4 or later, or set moves.jump_cuts to hard or punch.")
    }
} catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(2)
}
