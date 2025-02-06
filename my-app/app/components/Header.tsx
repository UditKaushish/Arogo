import { Stethoscope } from "lucide-react"

export default function Header() {
  return (
    <header className="mb-4 flex items-center space-x-2">
      <Stethoscope className="w-8 h-8 text-blue-500" />
      <div>
        <h1 className="text-2xl font-bold text-blue-600 dark:text-blue-400">MediChat AI Demo</h1>
        <p className="text-sm text-gray-600 dark:text-gray-300">
          Your AI-powered medical assistant demo. Ask any health-related questions for instant answers.
        </p>
      </div>
    </header>
  )
}

