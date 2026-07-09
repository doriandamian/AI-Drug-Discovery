import { CommonModule } from '@angular/common';
import { Component, OnDestroy, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MarkdownComponent } from 'ngx-markdown';

declare var SmilesDrawer: any;

const BACKEND_URL = 'http://localhost:8000';

interface ChatMessage {
  text: string;
  isAi: boolean;
  smilesList?: string[];
  status?: string;
  isStreaming?: boolean;
  elapsed?: string;
}

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule, MarkdownComponent],
  templateUrl: './chat.component.html',
  styleUrl: './chat.component.css',
})
export class ChatComponent implements OnInit, OnDestroy {
  messages: ChatMessage[] = [];
  userInput: string = '';
  isLoading: boolean = false;

  private tokenBuffer = '';
  private flushTimer?: ReturnType<typeof setTimeout>;
  private activeAiMsg?: ChatMessage;

  backendOnline: boolean | null = null;
  private healthTimer?: ReturnType<typeof setInterval>;

  readonly suggestions: string[] = [
    'Design a novel molecule for a given target',
    'Predict the toxicity of aspirin',
    'What is the molecular weight of caffeine?',
    'Find recent literature on kinase inhibitors',
  ];

  private readonly taskLabels: Record<string, string> = {
    cheminformatics_agent: 'Consulting the cheminformatics agent',
    safety_agent: 'Consulting the toxicology agent',
    literature_agent: 'Consulting the literature agent',
    graph_agent: 'Consulting the knowledge-graph agent',
    molecular_design_agent: 'Consulting the molecular design agent',
    search_pubmed: 'Searching PubMed',
    search_semantic_scholar: 'Searching Semantic Scholar',
    search_literature: 'Searching the local knowledge base',
    fetch_pubchem_properties: 'Fetching data from PubChem',
    predict_toxicity: 'Predicting toxicity',
    validate_smiles: 'Validating the structure',
    calculate_properties: 'Calculating properties',
    query_knowledge_graph: 'Querying the knowledge graph',
    enrich_drug_graph: 'Enriching the knowledge graph',
  };

  ngOnInit() {
    this.checkBackendHealth();
    this.healthTimer = setInterval(() => this.checkBackendHealth(), 8000);
  }

  ngOnDestroy() {
    if (this.healthTimer) clearInterval(this.healthTimer);
  }

  private async checkBackendHealth() {
    try {
      const res = await fetch(`${BACKEND_URL}/`, { method: 'GET' });
      this.backendOnline = res.ok;
    } catch {
      this.backendOnline = false;
    }
  }

  useSuggestion(text: string) {
    this.userInput = text;
    this.sendMessage();
  }

  async sendMessage() {
    if (!this.userInput.trim() || this.isLoading) return;

    const userText = this.userInput;
    this.messages.push({ text: userText, isAi: false });
    this.userInput = '';
    this.isLoading = true;

    const aiMsg: ChatMessage = { text: '', isAi: true, status: 'Connecting…', isStreaming: true };
    this.messages.push(aiMsg);
    this.activeAiMsg = aiMsg;
    this.tokenBuffer = '';

    try {
      const response = await fetch(`${BACKEND_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userText }),
      });

      if (!response.ok || !response.body) {
        throw new Error(`HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        const frames = buffer.split('\n\n');
        buffer = frames.pop() ?? '';

        for (const frame of frames) {
          const line = frame.trim();
          if (!line.startsWith('data:')) continue;
          const payload = line.slice(5).trim();
          if (!payload) continue;
          try {
            this.handleEvent(JSON.parse(payload), aiMsg);
          } catch {
          }
        }
      }
    } catch {
      this.flushTokenBuffer();
      aiMsg.status = undefined;
      this.backendOnline = false;
      if (!aiMsg.text) aiMsg.text = 'Failed to connect to the server. Please check that the backend is online.';
    } finally {
      this.flushTokenBuffer();
      this.activeAiMsg = undefined;
      this.isLoading = false;
      aiMsg.status = undefined;
    }
  }

  private isAgent(name: string): boolean {
    return name.endsWith('_agent');
  }

  private label(name: string): string {
    if (this.taskLabels[name]) return this.taskLabels[name];
    const pretty = name.replace(/_agent$/, '').replace(/_/g, ' ');
    return this.isAgent(name) ? `Consulting the ${pretty} agent` : `Running ${pretty}`;
  }

  private handleEvent(evt: any, aiMsg: ChatMessage) {
    switch (evt?.type) {
      case 'tool_call': {
        this.flushTokenBuffer();
        aiMsg.text = '';
        const tools: string[] = evt.tools ?? [];
        const labels = tools.map((t) => this.label(t));
        aiMsg.status = labels.join(', ') + '…';
        break;
      }
      case 'tool_result':
        aiMsg.status = 'Done: ' + this.label(evt.name);
        break;
      case 'reset': {
        if (this.flushTimer) {
          clearTimeout(this.flushTimer);
          this.flushTimer = undefined;
        }
        this.tokenBuffer = '';
        aiMsg.text = '';
        break;
      }
      case 'token':
        aiMsg.status = undefined;
        this.tokenBuffer += evt.content;
        if (!this.flushTimer) {
          this.flushTimer = setTimeout(() => this.flushTokenBuffer(), 16);
        }
        break;
      case 'final':
        this.flushTokenBuffer();
        aiMsg.status = undefined;
        if (evt.time) aiMsg.elapsed = evt.time;
        this.finalizeAiMessage(aiMsg, evt.message);
        break;
      case 'error':
        aiMsg.status = undefined;
        aiMsg.text = 'Error: ' + (evt.detail ?? 'unknown');
        break;
    }
  }

  private flushTokenBuffer() {
    if (this.flushTimer) {
      clearTimeout(this.flushTimer);
      this.flushTimer = undefined;
    }
    if (this.tokenBuffer && this.activeAiMsg) {
      this.activeAiMsg.text += this.tokenBuffer;
      this.tokenBuffer = '';
    }
  }

  private finalizeAiMessage(aiMsg: ChatMessage, finalText: string) {
    const source = finalText || aiMsg.text;

    const smilesRegex = /<smiles>(.*?)<\/smiles>/g;
    const extractedSmiles: string[] = [];
    let match;
    while ((match = smilesRegex.exec(source)) !== null) {
      const smilesCode = match[1].trim();
      if (smilesCode) extractedSmiles.push(smilesCode);
    }

    aiMsg.isStreaming = false;
    aiMsg.text = source.replace(
      /<smiles>(.*?)<\/smiles>/g,
      '\n**SMILES structure:** `$1`\n',
    );
    aiMsg.smilesList = extractedSmiles;

    if (extractedSmiles.length > 0) {
      const msgIndex = this.messages.indexOf(aiMsg);
      setTimeout(() => this.drawMolecules(extractedSmiles, msgIndex), 250);
    }
  }

  private drawMolecules(smilesList: string[], msgIndex: number) {
    if (typeof SmilesDrawer === 'undefined') {
      console.error('Error: the SmilesDrawer script is not available globally!');
      return;
    }

    const smilesDrawer = new SmilesDrawer.Drawer({ width: 300, height: 300 });

    smilesList.forEach((smileStr, smileIndex) => {
      const canvasId = `smiles-canvas-${msgIndex}-${smileIndex}`;
      SmilesDrawer.parse(
        smileStr,
        (tree: any) => {
          try {
            smilesDrawer.draw(tree, canvasId, 'light', false);
          } catch (drawError) {
            console.error(`Failed to render molecule [${smileStr}]:`, drawError);
          }
        },
        (parseError: any) => {
          console.error(
            `Invalid molecule structure [${smileStr}]:`,
            parseError,
          );
        },
      );
    });
  }
}
