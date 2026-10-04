import { mutateProps } from '../../src/mutations/mutationEngine.js';
import { MUTATION_LEVELS } from '../../src/mutations/mutationPresets.js';

console.log('=== Running Mutation Engine Unit Tests ===');

let testsPassed = 0;
let testsFailed = 0;

function assert(condition, message) {
  if (condition) {
    console.log(`✓ PASS: ${message}`);
    testsPassed++;
  } else {
    console.error(`✗ FAIL: ${message}`);
    testsFailed++;
  }
}

// 1. Level 0 Test (Original Behavior Unchanged)
const origProps = { id: 'username', placeholder: 'Username', type: 'text' };
const resL0 = mutateProps('username-input', origProps, 0, 42);
assert(resL0.id === 'username', 'Level 0 preserves original ID');
assert(resL0.placeholder === 'Username', 'Level 0 preserves original placeholder');
assert(resL0['data-semantic-role'] === 'username-input', 'Level 0 includes ground truth data-semantic-role');

// 2. Determinism Test (Same Level + Seed => Identical Output)
const resRun1 = mutateProps('username-input', origProps, 5, 42);
const resRun2 = mutateProps('username-input', origProps, 5, 42);
assert(resRun1.id === resRun2.id, 'Determinism: Same seed yields identical mutated ID');
assert(resRun1.placeholder === resRun2.placeholder, 'Determinism: Same seed yields identical mutated placeholder');

// 3. Seed Variation Test (Different Seed => Different Output)
const resSeed43 = mutateProps('username-input', origProps, 5, 43);
assert(resRun1.id !== resSeed43.id || resRun1.placeholder !== resSeed43.placeholder, 'Variation: Different seed yields different mutation');

// 4. Ground-Truth Invariant Preservation Test
const resL6 = mutateProps('password-input', { id: 'password', placeholder: 'Password' }, 6, 99);
assert(resL6['data-semantic-role'] === 'password-input', 'Ground-Truth Invariant: data-semantic-role remains unchanged in Level 6');
assert(resL6.id !== 'password', 'Level 6 mutates password ID');

// 5. Level 6 Held-Out Flag Check
assert(MUTATION_LEVELS[6].isHeldOut === true, 'Level 6 is explicitly flagged as isHeldOut: true');

console.log(`\nTests Summary: ${testsPassed} passed, ${testsFailed} failed.`);

if (testsFailed > 0) {
  process.exit(1);
}
