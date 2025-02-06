import { type FormEvent, type ChangeEvent, useState } from "react"
import { Send, Mic } from "lucide-react"

interface InputProps {
  input: string
  handleInputChange: (e: ChangeEvent<HTMLInputElement>) => void
  handleSubmit: (e: FormEvent<HTMLFormElement>) => void
}

export default function Input({ input, handleInputChange, handleSubmit }: InputProps) {
  const [isListening, setIsListening] = useState(false)

  const handleVoiceInput = () => {
    if ("webkitSpeechRecognition" in window) {
      const recognition = new (window as any).webkitSpeechRecognition()
      recognition.onstart = () => setIsListening(true)
      recognition.onend = () => setIsListening(false)
      recognition.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript
        handleInputChange({ target: { value: transcript } } as ChangeEvent<HTMLInputElement>)
      }
      recognition.start()
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex space-x-2">
      <input
        type="text"
        value={input}
        onChange={handleInputChange}
        className="flex-grow p-2 border rounded-lg dark:bg-gray-700 dark:border-gray-600 dark:text-white"
        placeholder="Type your health question..."
      />
      <button
        type="button"
        onClick={handleVoiceInput}
        className="p-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
      >
        <Mic className={`w-5 h-5 ${isListening ? "animate-pulse" : ""}`} />
      </button>
      <button type="submit" className="p-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600">
        <Send className="w-5 h-5" />
      </button>
    </form>
  )
}

