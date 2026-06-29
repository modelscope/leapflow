// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "OSHost",
    platforms: [
        .macOS(.v14),
    ],
    products: [
        .executable(name: "OSHost", targets: ["OSHost"]),
    ],
    targets: [
        .executableTarget(
            name: "OSHost",
            path: "Sources/OSHost"
        ),
    ]
)
