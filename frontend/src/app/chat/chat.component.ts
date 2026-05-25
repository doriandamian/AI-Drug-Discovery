import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MarkdownComponent } from 'ngx-markdown';

declare var SmilesDrawer: any;

interface ChatMessage {
  text: string;
  isAi: boolean;
  smilesList?: string[];
}

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule, MarkdownComponent],
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

    const userText = this.userInput;
    this.messages.push({ text: userText, isAi: false });
    this.userInput = '';
    this.isLoading = true;

    this.http
      .post<any>('http://localhost:8000/api/chat', { message: userText })
      .subscribe({
        next: (res) => {
          this.isLoading = false;
          this.processAiResponse(res.message);
        },
        error: (err) => {
          this.isLoading = false;
          this.messages.push({
            text: 'Eroare la conectarea cu serverul.',
            isAi: true,
          });
        },
      });
  }

  // Funcția nouă care procesează textul și desenează moleculele
  processAiResponse(aiText: string) {
    // Păstrăm textul curat, exact cum vine de la backend
    const formattedText = aiText;

    const smilesRegex = /<smiles>(.*?)<\/smiles>/g;
    let match;
    const extractedSmiles: string[] = [];

    // Extragem doar ce este prins clar între tag-uri explictie
    while ((match = smilesRegex.exec(formattedText)) !== null) {
      // Curățăm eventuale spații accidentale din tag
      const smilesCode = match[1].trim();
      if (smilesCode) {
        extractedSmiles.push(smilesCode);
      }
    }

    // Înlocuim tag-urile în textul final pentru un aspect vizual plăcut sub formă de cod
    const cleanText = formattedText.replace(
      /<smiles>(.*?)<\/smiles>/g,
      '\n**SMILES Structură:** `$1`\n',
    );

    this.messages.push({
      text: cleanText,
      isAi: true,
      smilesList: extractedSmiles,
    });

    if (extractedSmiles.length > 0) {
      setTimeout(() => {
        if (typeof SmilesDrawer === 'undefined') {
          console.error(
            'Eroare: Scriptul SmilesDrawer nu este disponibil global!',
          );
          return;
        }

        const smilesDrawer = new SmilesDrawer.Drawer({
          width: 300,
          height: 300,
        });
        const msgIndex = this.messages.length - 1;

        extractedSmiles.forEach((smileStr, smileIndex) => {
          const canvasId = `smiles-canvas-${msgIndex}-${smileIndex}`;

          SmilesDrawer.parse(
            smileStr,
            (tree: any) => {
              try {
                smilesDrawer.draw(tree, canvasId, 'light', false);
                console.log(`✅ Molecula [${smileStr}] a fost randată.`);
              } catch (drawError) {
                console.error(`❌ Eroare randare:`, drawError);
              }
            },
            (parseError: any) => {
              console.error(
                `❌ Structura nu este o moleculă validă [${smileStr}]:`,
                parseError,
              );
            },
          );
        });
      }, 250);
    }
  }
}
