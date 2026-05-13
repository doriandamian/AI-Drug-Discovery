import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';

interface ChatMessage {
  role: 'user' | 'ai' | 'error';
  text: string;
  time?: string;
}

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './chat.component.html',
  styleUrl: './chat.component.css',
})
export class ChatComponent {
  messages: ChatMessage[] = [];
  userInput: string = '';
  isLoading: boolean = false;

  constructor(private http: HttpClient) {}

  sendMessage() {
    if (!this.userInput.trim()) return;

    // 1. Adăugăm mesajul utilizatorului pe ecran
    const userMsg = this.userInput;
    this.messages.push({ role: 'user', text: userMsg });
    this.userInput = '';
    this.isLoading = true;

    // 2. Facem cererea (POST Request) către FastAPI (portul 8000)
    this.http
      .post<any>('http://localhost:8000/api/chat', { message: userMsg })
      .subscribe({
        next: (response) => {
          // 3. Adăugăm răspunsul Agentului AI
          this.messages.push({
            role: 'ai',
            text: response.message,
            time: response.time,
          });
          this.isLoading = false;
        },
        error: (error) => {
          console.error('API Error:', error);
          this.messages.push({
            role: 'error',
            text: 'Eroare de conexiune cu serverul AI. Asigură-te că Docker-ul cu FastAPI rulează.',
          });
          this.isLoading = false;
        },
      });
  }
}
