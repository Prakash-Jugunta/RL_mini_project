import React, { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { PRODUCTS } from '../data/seedData';
import { useMutation } from '../mutations/useMutation.js';

// GROUND-TRUTH METADATA ONLY.
//
// data-semantic-role is used by the environment/evaluator to determine
// the intended UI element and calculate rewards.
//
// It MUST NOT be included in the RL agent's observation/state features.
// Exposing this value to the agent would leak the correct answer.

export default function ProductDetailPage({ onAddToCart }) {
  const { productId } = useParams();
  const navigate = useNavigate();
  const [addedNotice, setAddedNotice] = useState(false);

  const { mutate } = useMutation();

  const product = PRODUCTS.find((p) => p.id === productId) || PRODUCTS[0];

  const handleAddToCart = () => {
    onAddToCart(product);
    setAddedNotice(true);
  };

  const addToCartProps = mutate('add-cart-action', {
    id: 'add-to-cart-btn',
    text: 'Add to Cart',
    className: 'btn btn-primary btn-block'
  });

  return (
    <div className="card" style={{ maxWidth: '640px', margin: '2rem auto' }}>
      <button
        className="btn btn-secondary"
        onClick={() => navigate('/products')}
        style={{ marginBottom: '1.5rem', padding: '0.4rem 0.8rem', fontSize: '0.85rem' }}
      >
        &larr; Back to Products
      </button>

      <div className="page-header">
        <h1 className="page-title">{product.name}</h1>
        <p className="page-subtitle">Product ID: {product.id}</p>
      </div>

      <p style={{ color: 'var(--text-muted)', marginBottom: '1.5rem', fontSize: '1rem', lineHeight: '1.6' }}>
        {product.description}
      </p>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
        <span style={{ fontSize: '1.75rem', fontWeight: 700, color: 'var(--primary-color)' }}>
          ${(product.price / 100).toFixed(2)}
        </span>
        <span style={{ backgroundColor: 'rgba(16, 185, 129, 0.2)', color: '#34d399', padding: '0.25rem 0.75rem', borderRadius: '9999px', fontSize: '0.875rem', fontWeight: 600 }}>
          In Stock (Deterministic)
        </span>
      </div>

      {addedNotice && (
        <div className="alert-success" style={{ marginBottom: '1.5rem' }}>
          Added {product.name} to cart!
        </div>
      )}

      {addToCartProps.renderAs === 'a' ? (
        <a
          id={addToCartProps.id}
          href="#add"
          role="button"
          aria-label="Add to Cart"
          data-semantic-role="add-cart-action"
          onClick={(e) => { e.preventDefault(); handleAddToCart(); }}
          className={addToCartProps.className}
          style={{ textDecoration: 'none', textAlign: 'center', display: 'block' }}
        >
          {addToCartProps.text || 'Add to Cart'}
        </a>
      ) : (
        <button
          id={addToCartProps.id}
          aria-label="Add to Cart"
          data-semantic-role="add-cart-action"
          onClick={handleAddToCart}
          className={addToCartProps.className}
        >
          {addToCartProps.text || 'Add to Cart'}
        </button>
      )}
    </div>
  );
}

