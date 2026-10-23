import Foundation
import ImageIO
import Vision

// Emit one JSON record per image. Diagnostics belong on stderr.
func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(2)
}

func emit(_ record: [String: Any]) throws {
    let data = try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys])
    print(String(decoding: data, as: UTF8.self))
}

func angle(_ value: NSNumber?) -> Any {
    if let value { return value.doubleValue }
    return NSNull()
}

func openness(_ region: VNFaceLandmarkRegion2D?, faceWidth: Double, faceHeight: Double) -> Any {
    guard let region, region.pointCount > 0 else { return NSNull() }
    var minX = Double.infinity
    var maxX = -Double.infinity
    var minY = Double.infinity
    var maxY = -Double.infinity
    for index in 0..<region.pointCount {
        let point = region.normalizedPoints[index]
        minX = min(minX, Double(point.x))
        maxX = max(maxX, Double(point.x))
        minY = min(minY, Double(point.y))
        maxY = max(maxY, Double(point.y))
    }
    let width = (maxX - minX) * faceWidth
    guard width > 0 else { return NSNull() }
    return (maxY - minY) * faceHeight / width
}

func measure(_ image: String) throws -> [String: Any] {
    let url = URL(fileURLWithPath: image)
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let picture = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw NSError(domain: "face-landmarks", code: 2,
                      userInfo: [NSLocalizedDescriptionKey: "Cannot read image: \(image)"])
    }
    let width = picture.width
    let height = picture.height
    let request = VNDetectFaceLandmarksRequest()
    let handler = VNImageRequestHandler(cgImage: picture, options: [:])
    try handler.perform([request])
    let faces: [[String: Any]] = (request.results ?? []).map { face in
        let box = face.boundingBox
        let x = max(0, min(Double(width), Double(box.minX) * Double(width)))
        let y = max(0, min(Double(height), (1 - Double(box.maxY)) * Double(height)))
        let right = max(x, min(Double(width), Double(box.maxX) * Double(width)))
        let bottom = max(y, min(Double(height), (1 - Double(box.minY)) * Double(height)))
        return [
            "x": x, "y": y, "w": right - x, "h": bottom - y,
            "left_eye": openness(face.landmarks?.leftEye, faceWidth: right - x,
                                 faceHeight: bottom - y),
            "right_eye": openness(face.landmarks?.rightEye, faceWidth: right - x,
                                  faceHeight: bottom - y),
            "yaw": angle(face.yaw), "roll": angle(face.roll),
            "confidence": Double(face.confidence),
        ]
    }
    return ["image": image, "width": width, "height": height, "faces": faces]
}

let images = Array(CommandLine.arguments.dropFirst())
if images.isEmpty { fail("Usage: face-landmarks IMAGE...") }
for image in images {
    do {
        try emit(measure(image))
    } catch {
        fail("Cannot measure image \(image): \(error.localizedDescription)")
    }
}
