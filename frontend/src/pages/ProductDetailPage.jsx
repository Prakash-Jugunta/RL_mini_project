import React, { useState } from 'react';
import { Link, useParams } from 'react-router-dom';
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
  const [addedProductId, setAddedProductId] = useState(null);

  const { mutate } = useMutation();

  const product = PRODUCTS.find((p) => p.id === productId);

  if (!product) {
    return (
      <section className="storefront-empty">
        <h1>Product not found</h1>
        <Link className="btn btn-primary" to="/products">Back to products</Link>
      </section>
    );
  }

  const handleAddToCart = () => {
    onAddToCart(product);
    setAddedProductId(product.id);
  };

  const addToCartProps = mutate('add-cart-action', {
    id: 'add-to-cart-btn',
    text: 'Add to Cart',
    className: 'btn btn-primary btn-block'
  });

  return (
    <div className="storefront-detail">
      <Link className="storefront-back" to="/products">&larr; Back to products</Link>
      <div className="storefront-detail-layout">
        <div className="storefront-product-media">
          <img src={`/storefront/${product.image}.webp`} alt={product.name} width="480" height="480" />
        </div>
        <div className="storefront-detail-copy">
          <p className="storefront-product-category">{product.category}</p>
          <h1>{product.name}</h1>
          <p className="storefront-detail-description">{product.description}</p>
          <dl className="storefront-specs">
            <div><dt>SKU</dt><dd>{product.sku}</dd></div>
            <div><dt>Availability</dt><dd>In stock</dd></div>
          </dl>
          <p className="product-price">${(product.price / 100).toFixed(2)}</p>

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
          {addedProductId === product.id && (
            <p className="alert-success" role="status">Added {product.name} to cart!</p>
          )}
        </div>
      </div>
    </div>
  );
}

