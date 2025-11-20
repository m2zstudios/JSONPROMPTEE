// backend/server.js
// Node 18+ recommended
import express from "express";
import rateLimit from "express-rate-limit";
import helmet from "helmet";
import cors from "cors";
import bodyParser from "body-parser";
import fetch from "node-fetch"; // if using node <18, otherwise global fetch can be used
import Ajv from "ajv";

const app = express();
app.use(helmet());
app.use(cors({
  origin: ["http://localhost:3000","http://127.0.0.1:3000"] // change to your frontend origin
}));
app.use(bodyParser.json({ limit: "8kb" }));

// simple rate-limiter
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 60, // tune as needed
});
app.use(limiter);

// Config via env
const OPENAI_KEY = process.env.BACKEND_OPENAI_KEY || process.env.OPENAI_API_KEY;

// JSON output schema (Ajv validator) - adjust fields as required
const ajv = new Ajv();
const jsonSchema = {
  type: "object",
  properties: {
    prompt: { type: "string" },
    negative_prompt: { type: "string" },
    style: { type: "string" },
    lighting: { type: "string" },
    camera: { type: "string" },
    details: {
      type: "object",
      properties: {
        subject: { type: "string" },
        background: { type: "string" },
        mood: { type: "string" },
        colors: { type: "string" }
      },
      required: ["subject"]
    },
    params: {
      type: "object",
      properties: {
        engine: { type: "string" },
        resolution: { type: "string" },
        cfg_scale: { type: "number" },
        steps: { type: "number" },
        sampler: { type: "string" },
        seed: { type: ["number","null"] }
      }
    }
  },
  required: ["prompt","params"]
};
const validateJSON = ajv.compile(jsonSchema);

// Utility to sanitize/limit input length
function sanitizeInput(text){
  if(!text || typeof text !== "string") return "";
  // limit to reasonable length
  const trimmed = text.trim().slice(0, 800); // tune max length
  return trimmed;
}

// System prompt to instruct the model to ONLY output JSON conforming to schema.
// Very important: make it explicit, no extra commentary.
const SYSTEM_PROMPT = `
You are a JSON generator. The user gives a natural-language image prompt. 
Your job: produce ONLY a single valid JSON object (no surrounding markdown, no explanation),
matching this structure:

{
  "prompt": "<concise image generation prompt>",
  "negative_prompt": "<optional>",
  "style": "<style tags>",
  "lighting": "<lighting description>",
  "camera": "<camera description>",
  "details": {
    "subject": "<main subject>",
    "background": "<background>",
    "mood": "<mood>",
    "colors": "<color hints>"
  },
  "params": {
    "engine": "<stable|mid|dalle>",
    "resolution": "WIDTHxHEIGHT",
    "cfg_scale": <number>,
    "steps": <number>,
    "sampler": "<sampler name>",
    "seed": null
  }
}

Rules:
- MUST output only JSON object, valid and parseable.
- Use concise, comma-separated clauses in "prompt".
- If a field is unknown, set it to an empty string or null.
- Do NOT include any commentary, explanation, or trailing text.
`;

// Endpoint: convert prompt to JSON
app.post("/api/convert", async (req, res) => {
  try {
    const userPromptRaw = req.body.prompt;
    const enginePref = req.body.engine || "stable"; // optional
    const userPrompt = sanitizeInput(userPromptRaw);
    if(!userPrompt) return res.status(400).json({ error: "Empty prompt" });

    // Build the chat messages: system + user.
    const userMsg = `
User Prompt: ${userPrompt}

Preferences:
- engine: ${enginePref}
- goal: Convert to image-generation JSON for pipelines (Stable Diffusion / Midjourney / DALL·E)
`;

    // Call OpenAI Chat Completions (example using v1/chat/completions).
    if(!OPENAI_KEY){
      return res.status(500).json({ error: "Server misconfigured: missing OPENAI API key" });
    }

    // Compose request to OpenAI
    const openaiRes = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type":"application/json",
        "Authorization": `Bearer ${OPENAI_KEY}`
      },
      body: JSON.stringify({
        model: "gpt-4o-mini", // example — choose low-latency text-capable model. Change as needed.
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          { role: "user", content: userMsg }
        ],
        temperature: 0.0,
        max_tokens: 500
      })
    });

    if(!openaiRes.ok){
      const txt = await openaiRes.text();
      console.error("OpenAI error:", txt);
      return res.status(502).json({ error: "AI provider error", detail: txt });
    }

    const data = await openaiRes.json();
    // Expect first choice content
    const rawText = data?.choices?.[0]?.message?.content || data?.choices?.[0]?.text;
    if(!rawText) return res.status(502).json({ error: "No response from AI provider" });

    // Try extracting JSON from the model output conservatively (strip surrounding text)
    const jsonCandidate = extractJSON(rawText);
    if(!jsonCandidate){
      return res.status(502).json({ error: "AI did not return JSON", raw: rawText });
    }

    // Parse JSON
    let outputObj;
    try {
      outputObj = JSON.parse(jsonCandidate);
    } catch(parseErr){
      console.error("JSON parse error:", parseErr, jsonCandidate);
      return res.status(502).json({ error: "AI returned invalid JSON", raw: jsonCandidate });
    }

    // Validate schema
    const valid = validateJSON(outputObj);
    if(!valid){
      console.error("Schema validation failed:", validateJSON.errors);
      return res.status(502).json({ error: "Generated JSON failed validation", details: validateJSON.errors, raw: outputObj });
    }

    // Enforce/override certain server policies (e.g., engine)
    outputObj.params.engine = enginePref;

    // Return sanitized json
    return res.json({ ok: true, result: outputObj });
  } catch (err) {
    console.error("Server error:", err);
    return res.status(500).json({ error: "Server error" });
  }
});

// Tiny helper: extract first JSON object from text
function extractJSON(text){
  // find first { ... } block (balanced brace)
  const start = text.indexOf('{');
  if(start === -1) return null;
  let depth = 0;
  let inString = false;
  for(let i = start; i < text.length; i++){
    const ch = text[i];
    if(ch === '"' && text[i-1] !== '\\') inString = !inString;
    if(!inString){
      if(ch === '{') depth++;
      else if(ch === '}') depth--;
      if(depth === 0){
        return text.slice(start, i+1);
      }
    }
  }
  return null;
}

const PORT = process.env.PORT || 4000;
app.listen(PORT, ()=> console.log("Server listening on", PORT));
