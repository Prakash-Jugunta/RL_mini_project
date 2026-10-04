import React from 'react';
import { Link } from 'react-router-dom';

// GROUND-TRUTH METADATA ONLY.
//
// data-semantic-role is used by the environment/evaluator to determine
// the intended UI element and calculate rewards.
//
// It MUST NOT be included in the RL agent's observation/state features.
// Exposing this value to the agent would leak the correct answer.

export default function OrderSuccessPage() {
  return (
    <div className="card" style={{ maxWidth: '540px', margin: '3rem auto', textAlign: 'center' }}>
      <div style={{ fontSize: '3rem', color: 'var(--success-color)', marginBottom: '1rem' }}>
        ✓
      </div>
      <h1 className="page-title">Thank You!</h1>
      
      <div
        id="order-success"
        className="alert-success"
        data-semantic-role="order-success-message"
        style={{ justifyContent: 'center', margin: '1.5rem 0', fontSize: '1.2rem' }}
      >
        Order placed successfully
      </div>

      <p style={{ color: 'var(--text-muted)', marginBottom: '2rem' }}>
        Your deterministic test order has been processed. Order confirmation #RL-DEMO-99823.
      </p>

      <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
        <Link to="/products" className="btn btn-primary">
          Back to Catalog
        </Link>
        <Link to="/dashboard" className="btn btn-secondary">
          Go to Dashboard
        </Link>
      </div>
    </div>
  );
}

