/**
 * Mutation Difficulty Levels, Preset Combinations, and Semantic Dictionaries
 *
 * GROUND-TRUTH METADATA AND EVALUATION POLICY WARNING:
 *
 * Level 5 defines KNOWN training combinations.
 * Level 6 defines HELD-OUT combinations — see explicit warning below.
 *
 * Each (level 5 or 6) seed maps deterministically to a named preset.
 * The same seed always selects the same preset regardless of run order.
 */

// ─── Level 5: KNOWN Combination Presets ─────────────────────────────────────
// Five distinct two-mutation pairings used for RL TRAINING.
// All are observable during training; none are held out.

export const LEVEL_5_PRESETS = {
  // seed → { name, activeMutations }
  11: { name: 'id+text',           activeMutations: ['id', 'text'] },
  22: { name: 'text+position',     activeMutations: ['text', 'position'] },
  33: { name: 'id+dom',            activeMutations: ['id', 'dom'] },
  44: { name: 'text+distractor',   activeMutations: ['text', 'distractor'] },
  55: { name: 'id+position',       activeMutations: ['id', 'position'] }
};

// ─── Level 6: HELD-OUT Combination Presets ──────────────────────────────────
//
//   ██████████████████████████████████████████████████████████████████
//   ██                                                              ██
//   ██   HELD OUT — NEVER USE FOR RL TRAINING                      ██
//   ██                                                              ██
//   ██   These presets are reserved for final evaluation only.      ██
//   ██   They must NOT appear in any training data, reward          ██
//   ██   calculation, or agent observation used during training.    ██
//   ██                                                              ██
//   ██████████████████████████████████████████████████████████████████
//
// Five distinct three-or-four-mutation combinations unseen during training.

export const LEVEL_6_PRESETS = {
  // seed → { name, activeMutations }
  11: { name: 'id+dom+distractor',           activeMutations: ['id', 'dom', 'distractor'] },
  22: { name: 'text+type+position',          activeMutations: ['text', 'type', 'position'] },
  33: { name: 'id+text+dom',                 activeMutations: ['id', 'text', 'dom'] },
  44: { name: 'position+dom+distractor',     activeMutations: ['position', 'dom', 'distractor'] },
  55: { name: 'id+text+dom+distractor',      activeMutations: ['id', 'text', 'dom', 'distractor'] }
};

/**
 * Resolve the active mutations for a given (level, seed).
 * For levels 5 and 6, seed selects a named preset.
 * For levels 0–4, activeMutations come directly from MUTATION_LEVELS.
 *
 * Returns: { name: string, activeMutations: string[] }
 */
export function resolvePreset(level, seed) {
  if (level === 5) {
    const preset = LEVEL_5_PRESETS[seed];
    if (!preset) {
      // Graceful fallback for unknown seeds — use id+text
      return { name: 'id+text (fallback)', activeMutations: ['id', 'text'] };
    }
    return preset;
  }
  if (level === 6) {
    const preset = LEVEL_6_PRESETS[seed];
    if (!preset) {
      return { name: 'id+dom+distractor (fallback)', activeMutations: ['id', 'dom', 'distractor'] };
    }
    return preset;
  }
  // Levels 0–4: fixed activeMutations, no seed-based branching
  const levelConfig = MUTATION_LEVELS[level] || MUTATION_LEVELS[0];
  return {
    name: levelConfig.name,
    activeMutations: levelConfig.activeMutations
  };
}

// ─── Base Mutation Level Definitions ────────────────────────────────────────

export const MUTATION_LEVELS = {
  0: {
    level: 0,
    name: 'Level 0 — Original',
    activeMutations: [],
    isHeldOut: false,
    description: 'Unmutated original baseline application'
  },
  1: {
    level: 1,
    name: 'Level 1 — ID + Class Mutation',
    activeMutations: ['id', 'class'],
    isHeldOut: false,
    description: 'ID and CSS class attribute mutations — directly breaks brittle #id selectors'
  },
  2: {
    level: 2,
    name: 'Level 2 — Text / Semantic Mutation',
    activeMutations: ['text'],
    isHeldOut: false,
    description: 'Text, label, placeholder, and button wording mutations'
  },
  3: {
    level: 3,
    name: 'Level 3 — Structural Mutation',
    activeMutations: ['position', 'dom'],
    isHeldOut: false,
    description: 'DOM nesting and element layout ordering mutations'
  },
  4: {
    level: 4,
    name: 'Level 4 — Distractor Mutation',
    activeMutations: ['distractor'],
    isHeldOut: false,
    description: 'Addition of realistic competing interactive elements'
  },
  5: {
    level: 5,
    name: 'Level 5 — Known Combined Mutations',
    // NOTE: activeMutations here are a PLACEHOLDER.
    // The real activeMutations for L5 come from resolvePreset(5, seed).
    activeMutations: ['id', 'text'],
    isHeldOut: false,
    description: 'Seed-mapped known training combinations (see LEVEL_5_PRESETS)'
  },
  6: {
    level: 6,
    name: 'Level 6 — HELD-OUT Combinations',
    // NOTE: activeMutations here are a PLACEHOLDER.
    // The real activeMutations for L6 come from resolvePreset(6, seed).
    activeMutations: ['id', 'text', 'dom', 'distractor'],
    isHeldOut: true,
    description: 'HELD OUT — NEVER USE FOR RL TRAINING. Seed-mapped unseen combinations.'
  }
};

/**
 * Predefined plausible software evolution semantic alternatives.
 * Preserves realistic UI vocabulary while altering string representation.
 */
export const SEMANTIC_DICTIONARY = {
  'username-input': {
    placeholder: ['Account', 'User ID', 'Account Name', 'Member Login'],
    label: ['Account Name', 'User ID', 'Login ID', 'User Identity']
  },
  'password-input': {
    placeholder: ['Passcode', 'Access Key', 'Secret', 'Security Code'],
    label: ['Passcode', 'Access Key', 'Account Password', 'Secret Key']
  },
  'login-action': {
    text: ['Continue', 'Sign In', 'Access Account', 'Proceed']
  },
  'search-input': {
    placeholder: ['Find products', 'Look Up', 'Search Catalog', 'Explore items']
  },
  'search-action': {
    text: ['Find', 'Look Up', 'Search Catalog', 'Explore']
  },
  'product-result': {
    // Dynamic suffix/prefix alternatives
  },
  'profile-name-input': {
    placeholder: ['Your Name', 'Account Holder', 'Member Name', 'Full Name Details'],
    label: ['Your Name', 'Account Holder', 'Member Name']
  },
  'profile-email-input': {
    placeholder: ['Contact Email', 'Electronic Mail', 'Account Email', 'Email Address'],
    label: ['Contact Email', 'User Email', 'Electronic Mail']
  },
  'profile-save-action': {
    text: ['Update Profile', 'Save Changes', 'Submit Profile', 'Store Info']
  },
  'add-cart-action': {
    text: ['Add to Bag', 'Put in Cart', 'Buy Item', 'Collect Item']
  },
  'cart-navigation-action': {
    text: ['Shopping Bag', 'My Cart', 'Selected Items', 'Basket']
  },
  'nav-cart': {
    text: ['Shopping Bag', 'My Cart', 'Selected Items', 'Basket']
  },
  'checkout-action': {
    text: ['Proceed to Checkout', 'Pay Now', 'Complete Order', 'Go to Checkout']
  },
  'confirm-order-action': {
    text: ['Place Order', 'Submit Order', 'Finalize Order', 'Complete Purchase']
  }
};

/**
 * Predefined realistic distractor templates for each page workflow.
 */
export const DISTRACTOR_TEMPLATES = {
  login: [
    {
      role: 'newsletter-email-input',
      type: 'input',
      idPrefix: 'newsletter-email',
      placeholder: 'Subscribe to Newsletter Email',
      label: 'Get Product Updates',
      inputType: 'text'
    },
    {
      role: 'create-account-action',
      type: 'button',
      idPrefix: 'create-account-btn',
      text: 'Create New Account',
      buttonType: 'button'
    }
  ],
  search: [
    {
      role: 'category-filter-select',
      type: 'select',
      idPrefix: 'category-filter',
      label: 'Category Filter',
      options: ['All Categories', 'Hardware', 'Peripherals', 'Accessories']
    },
    {
      role: 'header-newsletter-input',
      type: 'input',
      idPrefix: 'header-search',
      placeholder: 'Search Knowledge Base',
      inputType: 'text'
    }
  ],
  profile: [
    {
      role: 'secondary-phone-input',
      type: 'input',
      idPrefix: 'profile-phone',
      placeholder: 'Phone Number (Optional)',
      label: 'Phone Number',
      inputType: 'tel'
    },
    {
      role: 'reset-profile-action',
      type: 'button',
      idPrefix: 'reset-profile-btn',
      text: 'Reset Form',
      buttonType: 'button'
    }
  ],
  checkout: [
    {
      role: 'apply-coupon-input',
      type: 'input',
      idPrefix: 'coupon-code',
      placeholder: 'Enter Promo/Coupon Code',
      label: 'Discount Code',
      inputType: 'text'
    },
    {
      role: 'save-later-action',
      type: 'button',
      idPrefix: 'save-later-btn',
      text: 'Save Order for Later',
      buttonType: 'button'
    }
  ]
};
