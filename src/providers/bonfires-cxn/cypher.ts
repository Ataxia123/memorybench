// All Cypher used by the bonfires-cxn provider. Read-only, single group.
// Seed matching is exact-token: a term matches an entity iff it equals the
// whole lowercased name or one of its whitespace tokens (spec §0.5.2).

export const SEED_ENTITIES = `
MATCH (n:Entity {group_id: $groupId})
WHERE NOT n:Entity_Firing AND NOT n:Entity_Event AND NOT n:Entity_JointActivation
  AND any(t IN $terms WHERE toLower(n.name) = t OR t IN split(toLower(n.name), ' '))
WITH n
OPTIONAL MATCH (n)-[r]-()
WITH n, count(r) AS degree
RETURN n.uuid AS uuid, n.name AS name, degree
ORDER BY degree DESC, n.uuid ASC
LIMIT $maxSeeds`

export const SEEDS_TO_FIRINGS = `
UNWIND $seedUuids AS seedUuid
MATCH (s:Entity {uuid: seedUuid, group_id: $groupId})
MATCH (ev:Entity_Event {group_id: $groupId})-->(s)
MATCH (ev)-[:PART_OF]->(f:Entity_Firing {group_id: $groupId})
RETURN f.uuid AS uuid, f.name AS name, f.attributes AS attributes,
       collect(DISTINCT seedUuid) AS matchedSeeds`

export const NEIGHBORS_TO_FIRINGS = `
UNWIND $seedUuids AS seedUuid
MATCH (s:Entity {uuid: seedUuid, group_id: $groupId})
MATCH (s)-[r]-(o:Entity {group_id: $groupId})
WHERE NOT o:Entity_Firing AND NOT o:Entity_Event AND NOT o:Entity_JointActivation
  AND type(r) <> 'PART_OF' AND type(r) <> 'PART_OF_JOINT'
MATCH (ev:Entity_Event {group_id: $groupId})-->(o)
MATCH (ev)-[:PART_OF]->(f:Entity_Firing {group_id: $groupId})
RETURN f.uuid AS uuid, f.name AS name, f.attributes AS attributes,
       collect(DISTINCT seedUuid) AS matchedSeeds`

export const FIRING_STRUCTURES = `
UNWIND $firingUuids AS firingUuid
MATCH (ev:Entity_Event {group_id: $groupId})-[:PART_OF]->(f:Entity_Firing {uuid: firingUuid, group_id: $groupId})
MATCH (ev)-[r]->(x:Entity {group_id: $groupId})
WHERE type(r) <> 'PART_OF'
RETURN firingUuid, f.name AS firingName, ev.uuid AS eventUuid, ev.attributes AS eventAttributes,
       type(r) AS role, x.name AS filler
ORDER BY firingUuid ASC, eventUuid ASC, role ASC, filler ASC`
