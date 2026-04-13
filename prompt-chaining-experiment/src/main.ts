import "dotenv/config";
import OpenAI from "openai";
import {
  createPrompt,
  getPrompt,
  createDataset,
  getDataset,
  listDatasetExamples,
  createExperiment,
} from "@arizeai/ax-client";

const SPACE = process.env.ARIZE_SPACE_ID!;
const MODEL = "gpt-4o-mini";

const openai = new OpenAI();

// ── Inline dataset ──────────────────────────────────────────────────────────
const TOPICS = [
  "Why observability matters for LLM applications",
  "Getting started with prompt engineering",
  "Best practices for evaluating AI output quality",
  "How prompt chaining improves complex workflows",
  "The role of experiments in iterating on LLM systems",
];

// ── Helpers ─────────────────────────────────────────────────────────────────

function fillTemplate(template: string, vars: Record<string, string>): string {
  return template.replace(/\{(\w+)\}/g, (_, key) => vars[key] ?? `{${key}}`);
}

function isConflict(err: unknown): boolean {
  return (err as { statusCode?: number }).statusCode === 409;
}

async function callOpenAI(
  messages: { role: string; content: string }[],
): Promise<string> {
  const res = await openai.chat.completions.create({
    model: MODEL,
    messages: messages as OpenAI.ChatCompletionMessageParam[],
  });
  return res.choices[0].message.content ?? "";
}

// ── Main ────────────────────────────────────────────────────────────────────

async function main() {
  // Step 1 — Create Prompt 1 in Prompt Hub (or reuse existing)
  console.log("Step 1: Creating content-drafter prompt…");
  let drafterId: string;
  try {
    const drafter = await createPrompt({
      space: SPACE,
      name: "content-drafter",
      description: "Generates a raw blog introduction from a topic",
      version: {
        commitMessage: "Initial version",
        inputVariableFormat: "f_string",
        provider: "openAI",
        model: MODEL,
        messages: [
          {
            role: "system",
            content:
              "You are a skilled content writer. Write a short, engaging blog introduction (2-3 paragraphs) for the given topic.",
          },
          { role: "user", content: "Topic: {topic}" },
        ],
      },
    });
    drafterId = drafter.id;
    console.log(`  → Prompt created: ${drafterId}`);
  } catch (err) {
    if (!isConflict(err)) throw err;
    const existing = await getPrompt({ prompt: "content-drafter", space: SPACE });
    drafterId = existing.id;
    console.log(`  → Prompt already exists: ${drafterId}`);
  }

  // Step 2 — Create Prompt 2 in Prompt Hub (or reuse existing)
  console.log("Step 2: Creating content-refiner prompt…");
  let refinerId: string;
  try {
    const refiner = await createPrompt({
      space: SPACE,
      name: "content-refiner",
      description: "Polishes a draft for tone, clarity, and conciseness",
      version: {
        commitMessage: "Initial version",
        inputVariableFormat: "f_string",
        provider: "openAI",
        model: MODEL,
        messages: [
          {
            role: "system",
            content:
              "You are an expert editor. Refine the following draft for clarity, conciseness, and a professional tone. Return only the improved text.",
          },
          { role: "user", content: "{draft}" },
        ],
      },
    });
    refinerId = refiner.id;
    console.log(`  → Prompt created: ${refinerId}`);
  } catch (err) {
    if (!isConflict(err)) throw err;
    const existing = await getPrompt({ prompt: "content-refiner", space: SPACE });
    refinerId = existing.id;
    console.log(`  → Prompt already exists: ${refinerId}`);
  }

  // Step 3 — Create dataset with inline examples (or reuse existing)
  console.log("Step 3: Creating dataset…");
  let datasetId: string;
  try {
    const dataset = await createDataset({
      space: SPACE,
      name: "prompt-chaining-topics",
      examples: TOPICS.map((topic) => ({ topic })),
    });
    datasetId = dataset.id;
    console.log(`  → Dataset created: ${datasetId}`);
  } catch (err) {
    if (!isConflict(err)) throw err;
    const existing = await getDataset({ dataset: "prompt-chaining-topics", space: SPACE });
    datasetId = existing.id;
    console.log(`  → Dataset already exists: ${datasetId}`);
  }

  // Retrieve example IDs so we can map runs back to them
  const examples = await listDatasetExamples({ dataset: datasetId });

  // Step 4 — Pull both prompts back from Prompt Hub
  console.log("Step 4: Pulling prompts from Prompt Hub…");
  const drafterPrompt = await getPrompt({ prompt: drafterId });
  const refinerPrompt = await getPrompt({ prompt: refinerId });
  console.log("  → Prompts retrieved");

  const drafterMessages = drafterPrompt.version.messages;
  const refinerMessages = refinerPrompt.version.messages;

  // Step 5 — Run the chain for each example
  console.log("Step 5: Running prompt chain…");
  const runs: { exampleId: string; output: string; draft: string }[] = [];

  for (const example of examples) {
    const topic = example.topic as string;
    console.log(`  • "${topic}"`);

    // Prompt 1 → Draft
    const drafterFilled = drafterMessages.map((m) => ({
      role: m.role,
      content: fillTemplate(m.content ?? "", { topic }),
    }));
    const draft = await callOpenAI(drafterFilled);

    // Prompt 2 → Refined output
    const refinerFilled = refinerMessages.map((m) => ({
      role: m.role,
      content: fillTemplate(m.content ?? "", { draft }),
    }));
    const refined = await callOpenAI(refinerFilled);

    runs.push({ exampleId: example.id, output: refined, draft });
  }

  // Step 6 — Log experiment to Arize
  console.log("Step 6: Logging experiment…");
  const experiment = await createExperiment({
    experimentName: `prompt-chaining-${new Date().toISOString().replace(/[:.]/g, "-")}`,
    dataset: datasetId,
    experimentRuns: runs.map((r) => ({
      exampleId: r.exampleId,
      output: JSON.stringify({ refined: r.output, draft: r.draft }),
    })),
  });
  console.log(`  → Experiment created: ${experiment.id}`);
  console.log(
    "\nDone! Open Arize → Datasets & Experiments to view results and run evaluations.",
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
