import { useState } from "react"
import type { Message } from "ai"
import { VolumeIcon as VolumeUp } from "lucide-react"

export default function MessageComponent({ message }: { message: Message }) {
  const [isSpeaking, setIsSpeaking] = useState(false)

  const handleListen = () => {
    if ("speechSynthesis" in window) {
      const utterance = new SpeechSynthesisUtterance(message.content)
      utterance.onend = () => setIsSpeaking(false)
      setIsSpeaking(true)
      window.speechSynthesis.speak(utterance)
    }
  }

  return (
    <div className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-md p-4 rounded-lg ${
          message.role === "user"
            ? "bg-blue-500 text-white"
            : "bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200"
        }`}
      >
        <p>{message.content}</p>
        {message.role === "assistant" && (
          <button
            onClick={handleListen}
            disabled={isSpeaking}
            className="mt-2 text-sm text-blue-600 dark:text-blue-400 flex items-center"
          >
            <VolumeUp className="w-4 h-4 mr-1" />
            {isSpeaking ? "Speaking..." : "Listen"}
          </button>
        )}
      </div>
    </div>
  )
}

