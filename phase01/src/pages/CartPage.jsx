import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useMutation } from '../mutations/useMutation.js';

// GROUND-TRUTH METADATA ONLY.
//
// data-semantic-role is used by the environment/evaluator to determine
// the intended UI element and calculate rewards.
//
// It MUST NOT be included in the RL agent's observation/state features.
// Exposing this value to the agent would leak the correct answer.

export default function CartPage({ cartItems, onRemoveFromCart, onClearCart }) {
  const navigate = useNavigate();
  const { mutate } = useMutation();

  const calculateTotal = () => {
    return cartItems.reduce((acc, item) => acc + item.price * (item.quantity || 1), 0);
  };

  const checkoutBtnProps = mutate('checkout-action', {
    id: 'checkout-btn',
    text: 'Checkout',
    className: 'btn btn-primary'
  });

  const handleCheckoutClick = () => {
    navigate('/checkout');
  };

  return (
    <div className="card">
      <div className="page-header">
        <h1 className="page-title">Shopping Cart</h1>
        <p className="page-subtitle">Review items in your cart before checkout</p>
      </div>

      {cartItems.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '2rem 0' }}>
          <p style={{ color: 'var(--text-muted)', marginBottom: '1rem' }}>Your shopping cart is currently empty.</p>
          <button className="btn btn-primary" onClick={() => navigate('/products')}>
            Browse Products
          </button>
        </div>
      ) : (
        <>
          <table className="cart-table">
            <thead>
              <tr>
                <th>Product</th>
                <th>Price</th>
                <th>Quantity</th>
                <th>Subtotal</th>
              </tr>
            </thead>
            <tbody>
              {cartItems.map((item) => (
                <tr key={item.id}>
                  <td style={{ fontWeight: 600 }}>{item.name}</td>
                  <td>${(item.price / 100).toFixed(2)}</td>
                  <td>{item.quantity || 1}</td>
                  <td>${((item.price * (item.quantity || 1)) / 100).toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <div className="cart-summary">
            <span>Total:</span>
            <span style={{ color: 'var(--primary-color)' }}>${(calculateTotal() / 100).toFixed(2)}</span>
          </div>

          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '1rem', marginTop: '1.5rem' }}>
            <button className="btn btn-secondary" onClick={() => navigate('/products')}>
              Continue Shopping
            </button>
            <button
              id={checkoutBtnProps.id}
              aria-label="Checkout"
              data-semantic-role="checkout-action"
              className={checkoutBtnProps.className}
              onClick={handleCheckoutClick}
            >
              {checkoutBtnProps.text || 'Checkout'}
            </button>
          </div>
        </>
      )}
    </div>
  );
}

