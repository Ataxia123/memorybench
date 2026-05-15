import { BonfiresClient } from "../providers/bonfires/client.js"

const BONFIRE_ID = process.env.BONFIRE_ID ?? "69e6ae3482a950f64d3f5295"
const API_URL = process.env.BONFIRES_API_URL ?? "http://localhost:8000"
const API_KEY = process.env.BONFIRES_API_KEY ?? "local-dev-api-key"

const GENERIC_FALLBACK_TYPES = [
  {
    name: "Person",
    description:
      "A specific named individual — speaker, acquaintance, family member, public figure.",
  },
  {
    name: "Event",
    description: "A discrete named occurrence — conference, ceremony, trip, party, performance.",
  },
  {
    name: "Place",
    description: "A named physical location — city, neighborhood, venue, park, building, country.",
  },
  {
    name: "Organization",
    description: "A named collective — company, school, club, support group, nonprofit.",
  },
  {
    name: "Activity",
    description: "A named recurring pursuit or practice — running, painting, pottery, counseling.",
  },
  { name: "Object", description: "A specific distinguishable physical artifact." },
]

async function main() {
  const client = new BonfiresClient({ apiUrl: API_URL, apiKey: API_KEY })
  await client.healthz()

  const log = (msg: string) => console.log(`[${new Date().toISOString()}] ${msg}`)

  log(`bonfire=${BONFIRE_ID}`)

  log("1/5 startTaxonomy")
  const taxonomy = await client.startTaxonomy(BONFIRE_ID)
  log(`   job=${taxonomy.job_id}`)
  await client.waitForJob(taxonomy.job_id, { kind: "taxonomy", timeoutSec: 1800 })
  log("   ✓ taxonomy")

  log("2/5 buildGrammar (creates Ontology doc via derive_from_taxonomy)")
  await client.buildGrammar(BONFIRE_ID)
  log("   ✓ grammar (seed)")

  log("3/5 merge generic fallback types into ontology")
  const current = (await client.getOntology(BONFIRE_ID)) as {
    entity_labels: Array<{ name: string; description: string }>
  }
  const existingNames = new Set(current.entity_labels.map((l) => l.name))
  const merged = [
    ...current.entity_labels.map((l) => ({ name: l.name, description: l.description })),
    ...GENERIC_FALLBACK_TYPES.filter((g) => !existingNames.has(g.name)),
  ]
  await client.setOntology(BONFIRE_ID, merged)
  log(`   ✓ ontology (${merged.length} labels)`)

  log("4/5 buildCommunities (sync=true)")
  await client.buildCommunities(BONFIRE_ID)
  log("   ✓ communities")

  log(
    "5/5 buildOntology — cascade grammar (linkMethod=cosine, no ontology-labeled entities present)"
  )
  await client.buildOntology(BONFIRE_ID, {
    linkMethod: "cosine",
    extendGrammar: "locomo",
    grammarMinMentions: 1,
    grammarMinRelations: 1,
  })
  log("   ✓ cascade grammar built")

  log("DONE")
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})
