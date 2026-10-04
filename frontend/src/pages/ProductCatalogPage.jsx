import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { PRODUCTS } from '../data/seedData';
import { useMutation } from '../mutations/useMutation.js';

// GROUND-TRUTH METADATA ONLY.
//
// data-semantic-role is used by the environment/evaluator to determine
// the intended UI element and calculate rewards.
//
// It MUST NOT be included in the RL agent's observation/state features.
// Exposing this value to the agent would leak the correct answer.

export default function ProductCatalogPage({ onAddToCart }) {
  const [searchTerm, setSearchTerm] = useState('');
  const [activeQuery, setActiveQuery] = useState('');
  const navigate = useNavigate();

  const { mutate, getDistractors, shouldWrap } = useMutation();

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    setActiveQuery(searchTerm.trim());
  };

  const filteredProducts = PRODUCTS.filter((product) =>
    product.name.toLowerCase().includes(activeQuery.toLowerCase())
  );

  const handleProductClick = (productId) => {
    navigate(`/products/${productId}`);
  };

  const searchInputProps = mutate('search-input', {
    id: 'search-input',
    name: 'search',
    type: 'text',
    placeholder: 'Search products',
    className: 'form-input'
  });

  const searchBtnProps = mutate('search-action', {
    id: 'search-btn',
    type: 'submit',
    text: 'Search',
    className: 'btn btn-primary'
  });

  const distractors = getDistractors('search');

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Product Catalog</h1>
        <p className="page-subtitle">Search and explore available hardware products</p>
      </div>

      {distractors.map((d) => (
        <div key={d.id} style={{ marginBottom: '1rem' }}>
          {d.type === 'select' && (
            <div className="form-group">
              <label htmlFor={d.id} className="form-label">{d.label}</label>
              <select id={d.id} className="form-input" data-semantic-role={d.role}>
                {d.options.map((opt) => (
                  <option key={opt}>{opt}</option>
                ))}
              </select>
            </div>
          )}
          {d.type === 'input' && (
            <input
              id={d.id}
              type="text"
              className="form-input"
              placeholder={d.placeholder}
              data-semantic-role={d.role}
              style={{ marginBottom: '1rem' }}
            />
          )}
        </div>
      ))}

      <form onSubmit={handleSearchSubmit} className="search-container">
        {shouldWrap('search-input') ? (
          <div className="dom-wrapper-x1" style={{ flex: 1 }}>
            <input
              id={searchInputProps.id}
              name={searchInputProps.name}
              type={searchInputProps.type}
              className={searchInputProps.className}
              placeholder={searchInputProps.placeholder}
              aria-label="Search products"
              data-semantic-role="search-input"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
        ) : (
          <input
            id={searchInputProps.id}
            name={searchInputProps.name}
            type={searchInputProps.type}
            className={searchInputProps.className}
            placeholder={searchInputProps.placeholder}
            aria-label="Search products"
            data-semantic-role="search-input"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        )}

        <button
          id={searchBtnProps.id}
          aria-label="Search"
          data-semantic-role="search-action"
          type="submit"
          className={searchBtnProps.className}
        >
          {searchBtnProps.text || 'Search'}
        </button>
      </form>

      <div className="product-grid" id="product-list">
        {filteredProducts.length > 0 ? (
          filteredProducts.map((product) => {
            const resultLinkProps = mutate('product-result', {
              id: `product-result-${product.id}`,
              text: product.name
            });
            const detailsBtnProps = mutate('product-details-action', {
              id: `view-details-${product.id}`,
              text: 'View Details',
              className: 'btn btn-secondary btn-block'
            });

            return (
              <div
                key={product.id}
                className="product-card"
                id={`product-card-${product.id}`}
              >
                <div>
                  <h2 className="product-title">
                    <Link
                      id={resultLinkProps.id}
                      data-semantic-role="product-result"
                      to={`/products/${product.id}`}
                      style={{ color: 'inherit', textDecoration: 'none' }}
                    >
                      {resultLinkProps.text || product.name}
                    </Link>
                  </h2>
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginBottom: '1rem' }}>
                    {product.description}
                  </p>
                </div>
                <div>
                  <div className="product-price">
                    ${(product.price / 100).toFixed(2)}
                  </div>
                  <button
                    id={detailsBtnProps.id}
                    data-semantic-role="product-details-action"
                    aria-label={`View Details ${product.name}`}
                    className={detailsBtnProps.className}
                    onClick={() => handleProductClick(product.id)}
                  >
                    {detailsBtnProps.text || 'View Details'}
                  </button>
                </div>
              </div>
            );
          })
        ) : (
          <div style={{ color: 'var(--text-muted)', gridColumn: '1 / -1', padding: '2rem 0' }}>
            No products found matching "{activeQuery}"
          </div>
        )}
      </div>
    </div>
  );
}

