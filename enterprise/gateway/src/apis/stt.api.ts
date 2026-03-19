import type { Nullable } from "@repo/types";
import OpenAI, { toFile } from "openai";

export type TranscribeInput = {
  audioBuffer: Buffer;
  mimeType: string;
  prompt?: Nullable<string>;
  language?: string;
};

export abstract class BaseSttApi {
  abstract transcribe(input: TranscribeInput): Promise<{ text: string }>;
  abstract pullModel(): Promise<{ done: boolean; error?: string }>;
}

export class GroqSttApi extends BaseSttApi {
  private client: OpenAI;
  private model: string;

  constructor(opts: { apiKey: string; model: string }) {
    super();
    this.client = new OpenAI({
      baseURL: "https://api.groq.com/openai/v1",
      apiKey: opts.apiKey,
    });
    this.model = opts.model;
  }

  async transcribe(input: TranscribeInput): Promise<{ text: string }> {
    const ext = input.mimeType.split("/").pop() ?? "wav";
    const file = await toFile(input.audioBuffer, `audio.${ext}`, {
      type: input.mimeType,
    });
    const result = await this.client.audio.transcriptions.create({
      file,
      model: this.model,
      prompt: input.prompt ?? undefined,
      language: input.language,
    });
    return { text: result.text };
  }

  async pullModel(): Promise<{ done: boolean; error?: string }> {
    return { done: true };
  }
}

export class SpeachesSttApi extends BaseSttApi {
  private client: OpenAI;
  private model: string;
  private baseURL: string;
  private apiKey: string;
  private cfAccessClientId: string;
  private cfAccessClientSecret: string;

  constructor(opts: {
    url: string;
    apiKey: string;
    model: string;
    cfAccessClientId?: string;
    cfAccessClientSecret?: string;
  }) {
    super();
    const url = `${opts.url}/v1`;
    this.cfAccessClientId = opts.cfAccessClientId ?? "";
    this.cfAccessClientSecret = opts.cfAccessClientSecret ?? "";
    this.client = new OpenAI({
      baseURL: url,
      apiKey: opts.apiKey,
      defaultHeaders: this.cfHeaders(),
    });
    this.model = opts.model;
    this.baseURL = url;
    this.apiKey = opts.apiKey;
  }

  private cfHeaders(): Record<string, string> {
    if (!this.cfAccessClientId || !this.cfAccessClientSecret) {
      return {};
    }
    return {
      "CF-Access-Client-Id": this.cfAccessClientId,
      "CF-Access-Client-Secret": this.cfAccessClientSecret,
    };
  }

  async transcribe(input: TranscribeInput): Promise<{ text: string }> {
    const ext = input.mimeType.split("/").pop() ?? "wav";
    const file = await toFile(input.audioBuffer, `audio.${ext}`, {
      type: input.mimeType,
    });
    const result = await this.client.audio.transcriptions.create({
      file,
      model: this.model,
      prompt: input.prompt ?? undefined,
      language: input.language,
    });
    return { text: result.text };
  }

  async pullModel(): Promise<{ done: boolean; error?: string }> {
    try {
      const res = await fetch(`${this.baseURL}/models/${this.model}`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${this.apiKey}`,
          ...this.cfHeaders(),
        },
      });
      if (res.ok) {
        return { done: true };
      }
      const text = await res.text().catch(() => res.statusText);
      return { done: false, error: text };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      return { done: false, error: message };
    }
  }
}
