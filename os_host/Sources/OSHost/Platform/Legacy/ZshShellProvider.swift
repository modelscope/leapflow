import Foundation

/// Shell execution provider using /bin/zsh.
struct ZshShellProvider: ShellProvider {
    func executeWithStatus(command: String) throws -> ShellResult {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/zsh")
        process.arguments = ["-lc", command]
        let outPipe = Pipe()
        let errPipe = Pipe()
        process.standardOutput = outPipe
        process.standardError = errPipe
        try process.run()
        process.waitUntilExit()
        let outData = outPipe.fileHandleForReading.readDataToEndOfFile()
        let errData = errPipe.fileHandleForReading.readDataToEndOfFile()
        let stdout = String(data: outData, encoding: .utf8) ?? ""
        let stderr = String(data: errData, encoding: .utf8) ?? ""
        let output = stderr.isEmpty ? stdout : stdout + "\n" + stderr
        return ShellResult(output: output, exitCode: process.terminationStatus)
    }
}
