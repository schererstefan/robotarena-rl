#!/usr/bin/env node
// Numeric verification helper: loads an exported weights JSON, normalizes a
// raw obs with the embedded VecNormalize stats, runs the MLP, prints logits.
// Used by rl/export_ts.py to check Python <-> Node agreement (< 1e-4).
// Usage: node verify_node.mjs <weights.json>   (reads obs JSON array on stdin)
// Output: {"logits": [...]}
import fs from 'node:fs';

const weightsPath = process.argv[2];
if (!weightsPath) {
    console.error('usage: node verify_node.mjs <weights.json>');
    process.exit(2);
}
const w = JSON.parse(fs.readFileSync(weightsPath, 'utf8'));

let stdin = '';
for await (const chunk of process.stdin) stdin += chunk;
const obs = JSON.parse(stdin);

function normalize(o) {
    const n = w.norm;
    if (!n) return o;
    return o.map((v, i) => {
        const x = (v - n.mean[i]) / Math.sqrt(n.var[i] + n.epsilon);
        return Math.min(n.clip_obs, Math.max(-n.clip_obs, x));
    });
}

function mlp(o) {
    let x = normalize(o);
    for (let li = 0; li < w.layers.length; li += 1) {
        const { W, b } = w.layers[li];
        const y = new Array(b.length);
        for (let j = 0; j < b.length; j += 1) {
            let acc = b[j];
            const row = W[j];
            for (let k = 0; k < x.length; k += 1) acc += row[k] * x[k];
            y[j] = acc;
        }
        const last = li === w.layers.length - 1;
        x = last ? y : y.map((v) => (w.activation === 'tanh' ? Math.tanh(v) : Math.max(0, v)));
    }
    return x;
}

process.stdout.write(JSON.stringify({ logits: mlp(obs) }) + '\n');
