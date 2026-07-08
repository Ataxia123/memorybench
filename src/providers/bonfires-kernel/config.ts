export interface KernelConfig {
  apiUrl: string
  bonfireId: string
  apiKey: string
  actorId: string
  expectedCardsDigest?: string
}

function required(env: Record<string, string | undefined>, key: string): string {
  const value = env[key]?.trim()
  if (!value) throw new Error(`bonfires-kernel: missing required env ${key}`)
  return value
}

export function loadKernelConfig(
  env: Record<string, string | undefined> = process.env
): KernelConfig {
  return {
    apiUrl: required(env, "KERNELB_API_URL"),
    bonfireId: required(env, "KERNELB_BONFIRE_ID"),
    apiKey: required(env, "KERNELB_API_KEY"),
    actorId: env.KERNELB_ACTOR_ID?.trim() || "bench",
    expectedCardsDigest: env.KERNELB_EXPECTED_CARDS_DIGEST?.trim() || undefined,
  }
}
