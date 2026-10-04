import React from 'react';
import { useNavigate, Navigate } from 'react-router-dom';
import { useMutation } from '../mutations/useMutation.js';

// GROUND-TRUTH METADATA ONLY.
//
// data-semantic-role is used by the environment/evaluator to determine
// the intended UI element and calculate rewards.
//
// It MUST NOT be included in the RL agent's observation/state features.
// Exposing this value to the agent would leak the correct answer.

export default function CheckoutPage({ cartItems, onCompleteOrder }) {
  const navigate = useNavigate();
  const { mutate, getDistractors } = useMutation();

  // FIX 3: Prevent Empty-Cart Checkout - Redirect immediately if cart is empty
  if (!cartItems || cartItems.length === 0) {
    return <Navigate to="/cart" replace />;
  }

  const handleConfirmOrder = (e) => {
    e.preventDefault();
    if (onCompleteOrder) {
      onCompleteOrder();
    }
    navigate('/order-success');
  };

  const calculateTotal = () => {
    return cartItems.reduce((acc, item) => acc + item.price * (item.quantity || 1), 0);
  };

  const confirmBtnProps = mutate('confirm-order-action', {
    id: 'confirm-order-btn',
    text: 'Confirm Order',
    className: 'btn btn-success btn-block'
  });

  const distractors = getDistractors('checkout');

  return (
    <div className="card" style={{ maxWidth: '600px', margin: '2rem auto' }}>
      <div className="page-header">
        <h1 className="page-title">Order Checkout</h1>
        <p className="page-subtitle">Confirm shipping details and finalize your order</p>
      </div>

      <form onSubmit={handleConfirmOrder}>
        {distractors.map((d) => (
          <div key={d.id} className="form-group" style={{ marginBottom: '1rem' }}>
            <label htmlFor={d.id} className="form-label" style={{ color: '#94a3b8' }}>{d.label}</label>
            {d.type === 'input' && (
              <input
                id={d.id}
                type="text"
                className="form-input"
                placeholder={d.placeholder}
                data-semantic-role={d.role}
              />
            )}
            {d.type === 'button' && (
              <button
                id={d.id}
                type="button"
                className="btn btn-secondary btn-block"
                data-semantic-role={d.role}
              >
                {d.text}
              </button>
            )}
          </div>
        ))}

        <div style={{ marginBottom: '1.5rem', backgroundColor: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: '0.5rem', border: '1px solid var(--card-border)' }}>
          <h3 style={{ fontSize: '1rem', marginBottom: '0.5rem' }}>Shipping Address (Deterministic Demo)</h3>
          <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>123 Research Way, Suite 404, Tech Park</p>
        </div>

        <div style={{ marginBottom: '1.5rem', backgroundColor: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: '0.5rem', border: '1px solid var(--card-border)' }}>
          <h3 style={{ fontSize: '1rem', marginBottom: '0.5rem' }}>Payment Method</h3>
          <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>Demo Credit Card ending in 4242</p>
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', margin: '1.5rem 0', fontWeight: 700, fontSize: '1.25rem' }}>
          <span>Order Total:</span>
          <span style={{ color: 'var(--primary-color)' }}>
            ${(calculateTotal() / 100).toFixed(2)}
          </span>
        </div>

        <button
          id={confirmBtnProps.id}
          aria-label="Confirm Order"
          data-semantic-role="confirm-order-action"
          type="submit"
          className={confirmBtnProps.className}
        >
          {confirmBtnProps.text || 'Confirm Order'}
        </button>
      </form>
    </div>
  );
}

