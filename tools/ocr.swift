import Foundation
import Vision
import AppKit

// prints the recognised text of each image path given on the command line, one
// block per image, lines ordered top to bottom. vision sometimes drops lines on a
// large, mostly empty screenshot, so the image is read whole and again in
// overlapping horizontal bands; for each text row the longest reading is kept
// (a band edge can cut a row, which gives a shorter partial reading).
let bandHeight = 360
let overlap = 120

func recognise(_ image: CGImage) -> [(y: Double, text: String)] {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = false
    try? VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
    return (request.results ?? []).compactMap { observation in
        guard let text = observation.topCandidates(1).first?.string else { return nil }
        // vision uses a bottom-left origin, flip to a fraction from the top
        return (1.0 - observation.boundingBox.midY, text)
    }
}

for path in CommandLine.arguments.dropFirst() {
    guard let image = NSImage(contentsOfFile: path),
          let full = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        print("=== \(path)\nERROR could not load")
        continue
    }
    var readings: [(y: Double, text: String)] = recognise(full).map { ($0.y * Double(full.height), $0.text) }
    var top = 0
    while top < full.height {
        let height = min(bandHeight, full.height - top)
        if let band = full.cropping(to: CGRect(x: 0, y: top, width: full.width, height: height)) {
            readings += recognise(band).map { (Double(top) + $0.y * Double(height), $0.text) }
        }
        if top + height >= full.height { break }
        top += bandHeight - overlap
    }
    readings.sort { $0.y < $1.y }
    var rows: [(y: Double, text: String)] = []
    for reading in readings {
        if let last = rows.last, abs(last.y - reading.y) < 10 {
            if reading.text.count > last.text.count { rows[rows.count - 1] = reading }
        } else {
            rows.append(reading)
        }
    }
    print("=== \(path)")
    print(rows.map { $0.text }.joined(separator: "\n"))
}
