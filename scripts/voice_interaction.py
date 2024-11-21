import speech_recognition as sr
import pyttsx3  # we really need to use a different library for this as pyttsx3 is not very realistic
import customtkinter as ctk

recognizer = sr.Recognizer()
tts = pyttsx3.init()


def say(text: str) -> None:
    tts.say(text)
    tts.runAndWait()


def listen() -> str:
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "Sorry, I did not understand that"
        except sr.RequestError:
            return "Sorry, my speech service is down"


class VoiceAssistantApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Mindful Voice Assistant")
        self.geometry("500x400")

        self.label = ctk.CTkLabel(self, text="Mindful", font=("Arial", 18))
        self.label.pack(pady=20)

        self.output_box = ctk.CTkTextbox(self, height=150, font=("Arial", 14))
        self.output_box.pack(pady=10, padx=10, fill="both", expand=True)

        self.listen_button = ctk.CTkButton(self, text="Speak", command=self.handle_listen, font=("Arial", 14))
        self.listen_button.pack(pady=10)

        self.exit_button = ctk.CTkButton(self, text="Exit", command=self.destroy, font=("Arial", 14), fg_color="red")
        self.exit_button.pack(pady=10)

    def handle_listen(self):
        self.output_box.insert("end", "\nListening...\n")
        user_input = listen()
        self.output_box.insert("end", f"User: {user_input}\n")

        if user_input.lower() in ["exit", "quit"]:
            self.output_box.insert("end", "Goodbye!\n")
            say("Goodbye!")
            self.destroy()
        else:
            response = f"You said: {user_input}"  # TODO: use llm instead
            self.output_box.insert("end", f"AI: {response}\n")
            say(response)


if __name__ == "__main__":
    app = VoiceAssistantApp()
    app.mainloop()
