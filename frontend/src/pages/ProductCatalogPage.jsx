import React, { useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { PRODUCTS } from '../data/seedData';
import { useMutation } from '../mutations/useMutation.js';

// GROUND-TRUTH METADATA ONLY.
//
// data-semantic-role is used by the environment/evaluator to determine
// the intended UI element and calculate rewards.
//
// It MUST NOT be included in the RL agent's observation/state features.
// Exposing this value to the agent would leak the correct answer.

const CATEGORIES = [...new Set(PRODUCTS.map((product) => product.category))];

export default function ProductCatalogPage() {
  const [searchTerm, setSearchTerm] = useState('');
  const [activeQuery, setActiveQuery] = useState('');
  const [category, setCategory] = useState('');
  const navigate = useNavigate();
  const { search } = useLocation();
  const fixture = import.meta.env.DEV ? new URLSearchParams(search).get('storefront-fixture') : null;
  const wrongProduct = import.meta.env.DEV && fixture === 'search-wrong-product';
  const searchInHeader = import.meta.env.DEV && (fixture === 'search-release-b' || wrongProduct);

  const { mutate, getDistractors, shouldWrap } = useMutation();

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    setActiveQuery(searchTerm.trim());
  };

  const query = activeQuery.toLowerCase();
  const filteredProducts = PRODUCTS.filter((product) =>
    (!category || product.category === category) &&
    (product.name.toLowerCase().includes(query) || product.sku.toLowerCase().includes(query))
  );

  const resetFilters = () => {
    setSearchTerm('');
    setActiveQuery('');
    setCategory('');
  };

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
    id: searchInHeader ? 'catalog-search-submit' : 'search-btn',
    type: 'submit',
    text: 'Search',
    className: 'btn btn-primary'
  });

  const distractors = getDistractors('search');

  const searchInput = (
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
  );

  const searchForm = (
    <form onSubmit={handleSearchSubmit} role="search" aria-label="Catalog" className="search-container">
      {shouldWrap('search-input') ? (
        <div className="dom-wrapper-x1">{searchInput}</div>
      ) : searchInput}
      <button
        id={searchBtnProps.id}
        aria-label="Search"
        data-semantic-role="search-action"
        type="submit"
        className={searchBtnProps.className}
      >
        {searchInHeader ? <span className="storefront-search-icon" aria-hidden="true" /> : searchBtnProps.text || 'Search'}
      </button>
    </form>
  );

  return (
    <div className="storefront-catalog">
      <header className="storefront-hero">
        <div className="storefront-hero-copy">
          <h1>Make room for<br />better work.</h1>
          <p>Keyboards, mice, and desk accessories for a workspace that feels like yours.</p>
          {searchInHeader ? searchForm : (
            <a className="btn btn-primary" href="#catalog">Shop the collection <span aria-hidden="true">↘</span></a>
          )}
        </div>
        <img src="/storefront/workspace.webp" alt="" width="900" height="600" fetchPriority="high" />
      </header>

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

      <section className="storefront-collection" aria-labelledby="catalog">
        <div className="storefront-collection-heading">
          <h2 id="catalog">Shop all gear</h2>
          <p className="storefront-result-count" role="status">
            {filteredProducts.length} {filteredProducts.length === 1 ? 'product' : 'products'}
          </p>
        </div>
        {!searchInHeader && searchForm}
        <div className="storefront-categories" aria-label="Product categories" role="group">
          <button type="button" aria-pressed={!category} onClick={() => setCategory('')}>All gear</button>
          {CATEGORIES.map((name) => (
            <button key={name} type="button" aria-pressed={category === name} onClick={() => setCategory(name)}>
              {name}
            </button>
          ))}
        </div>

        <div className="product-grid" id="product-list">
          {filteredProducts.length > 0 ? (
            filteredProducts.map((product) => {
              const productRoute = wrongProduct && product.id === 'wireless-mouse'
                ? 'wireless-mouse-pro'
                : product.id;
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
                <article
                  key={product.id}
                  className="product-card"
                  id={`product-card-${product.id}`}
                >
                  <div className="storefront-product-media">
                    <img src={`/storefront/${product.image}.webp`} alt="" width="480" height="480" loading="lazy" />
                  </div>
                  <div className="storefront-product-copy">
                    <p className="storefront-product-category">{product.category}</p>
                    <h3 className="product-title">
                      <Link
                        id={resultLinkProps.id}
                        data-semantic-role="product-result"
                        to={`/products/${productRoute}`}
                      >
                        {resultLinkProps.text || product.name}
                      </Link>
                    </h3>
                    <p className="storefront-product-sku">SKU {product.sku}</p>
                    <div className="storefront-product-bottom">
                      <span className="product-price">
                      ${(product.price / 100).toFixed(2)}
                      </span>
                      <button
                        id={detailsBtnProps.id}
                        data-semantic-role="product-details-action"
                        aria-label={`View Details ${product.name}`}
                        className={detailsBtnProps.className}
                        onClick={() => handleProductClick(productRoute)}
                      >
                        {detailsBtnProps.text || 'View Details'}
                      </button>
                    </div>
                  </div>
                </article>
              );
            })
          ) : (
            <div className="storefront-empty">
              <h3>No products found</h3>
              <p>{activeQuery ? `No results for “${activeQuery}”.` : 'No products match these filters.'}</p>
              <button type="button" className="btn btn-secondary" onClick={resetFilters}>Clear search and filters</button>
            </div>
          )}
        </div>
      </section>
      {searchInHeader && <HelpSearch />}
    </div>
  );
}

function HelpSearch() {
  const [query, setQuery] = useState('');
  const [submitted, setSubmitted] = useState(false);

  return (
    <aside className="storefront-collection" aria-labelledby="help-title">
      <h2 id="help-title">Product support</h2>
      <form
        role="search"
        aria-label="Help center"
        className="search-container"
        onSubmit={(event) => {
          event.preventDefault();
          setSubmitted(true);
        }}
      >
        <input
          className="form-input"
          aria-label="Search help articles"
          placeholder="Search help articles"
          value={query}
          onChange={(event) => { setQuery(event.target.value); setSubmitted(false); }}
          required
        />
        <button type="submit" className="btn btn-secondary">Search</button>
      </form>
      {submitted && <p role="status">No help articles found for “{query}”.</p>}
    </aside>
  );
}

