import AVFoundation
import CoreGraphics
import Foundation
import ImageIO

/// Extracts individual frames from recorded video files using AVAssetImageGenerator.
enum VideoFrameExtractor {

    /// Extracts a single frame from a video at the specified timestamp.
    /// - Parameters:
    ///   - videoPath: Absolute path to the .mp4 video file.
    ///   - timestampS: Timestamp in seconds from which to extract the frame.
    ///   - maxSize: Optional maximum dimension (width or height) for downscaling.
    /// - Returns: A tuple containing JPEG data, width, and height of the extracted frame.
    static func extractFrame(videoPath: String, timestampS: Double, maxSize: Int? = nil) async throws -> (data: Data, width: Int, height: Int) {
        let url = URL(fileURLWithPath: videoPath)
        guard FileManager.default.fileExists(atPath: videoPath) else {
            throw FrameExtractionError.fileNotFound(videoPath)
        }

        let asset = AVAsset(url: url)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = CMTime(seconds: 0.1, preferredTimescale: 600)
        generator.requestedTimeToleranceAfter = CMTime(seconds: 0.1, preferredTimescale: 600)

        if let maxSize = maxSize, maxSize > 0 {
            generator.maximumSize = CGSize(width: maxSize, height: maxSize)
        }

        let time = CMTime(seconds: timestampS, preferredTimescale: 600)
        let cgImage: CGImage

        if #available(macOS 13.0, *) {
            let (image, _) = try await generator.image(at: time)
            cgImage = image
        } else {
            cgImage = try generator.copyCGImage(at: time, actualTime: nil)
        }

        // Encode as JPEG
        let mutableData = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            mutableData,
            "public.jpeg" as CFString,
            1,
            nil
        ) else {
            throw FrameExtractionError.encodingFailed
        }
        let options: [CFString: Any] = [
            kCGImageDestinationLossyCompressionQuality: 0.8,
        ]
        CGImageDestinationAddImage(destination, cgImage, options as CFDictionary)
        guard CGImageDestinationFinalize(destination) else {
            throw FrameExtractionError.encodingFailed
        }

        return (data: mutableData as Data, width: cgImage.width, height: cgImage.height)
    }
}

// MARK: - Errors

enum FrameExtractionError: Error, CustomStringConvertible {
    case fileNotFound(String)
    case encodingFailed
    case invalidTimestamp

    var description: String {
        switch self {
        case .fileNotFound(let path):
            return "Video file not found: \(path)"
        case .encodingFailed:
            return "Failed to encode frame as JPEG"
        case .invalidTimestamp:
            return "Invalid timestamp for frame extraction"
        }
    }
}
