// Write a HEIC from a PNG with an orientation flag and Display P3 tags.
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

let args = CommandLine.arguments
let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: args[1]) as CFURL, nil)!
var image = CGImageSourceCreateImageAtIndex(source, 0, nil)!
if let p3 = CGColorSpace(name: CGColorSpace.displayP3), let tagged = image.copy(colorSpace: p3) {
    image = tagged
}
let target = CGImageDestinationCreateWithURL(
    URL(fileURLWithPath: args[2]) as CFURL, UTType.heic.identifier as CFString, 1, nil)!
CGImageDestinationAddImage(target, image, [kCGImagePropertyOrientation: Int(args[3])!] as CFDictionary)
if !CGImageDestinationFinalize(target) { exit(1) }
