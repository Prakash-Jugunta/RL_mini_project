import React from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useMutation } from '../mutations/useMutation.js';

// GROUND-TRUTH METADATA ONLY.
//
// data-semantic-role is used by the environment/evaluator to determine
// the intended UI element and calculate rewards.
//
// It MUST NOT be included in the RL agent's observation/state features.
// Exposing this value to the agent would leak the correct answer.

export default function Navbar({ cartCount, user, onLogout }) {
  const navigate = useNavigate();
  const location = useLocation();

  const { mutate } = useMutation();

  const handleCartClick = () => {
    navigate('/cart');
  };

  const navCartProps = mutate('nav-cart', {
    id: 'cart-btn',
    text: 'Cart',
    className: 'btn btn-secondary'
  });

  return (
    <nav className="navbar" role="navigation" aria-label="Main Navigation">
      <Link to="/" className="navbar-brand">
        RL <span>Self-Healing</span> Demo
      </Link>
      <ul className="navbar-nav">
        <li>
          <Link
            id="nav-products"
            aria-label="Products"
            data-semantic-role="nav-products"
            to="/products"
            className={`nav-link ${location.pathname.startsWith('/products') ? 'active' : ''}`}
          >
            Products
          </Link>
        </li>
        {user ? (
          <>
            <li>
              <Link
                id="nav-dashboard"
                aria-label="Dashboard"
                data-semantic-role="nav-dashboard"
                to="/dashboard"
                className={`nav-link ${location.pathname === '/dashboard' ? 'active' : ''}`}
              >
                Dashboard
              </Link>
            </li>
            <li>
              <Link
                id="nav-profile"
                aria-label="Profile"
                data-semantic-role="nav-profile"
                to="/profile"
                className={`nav-link ${location.pathname === '/profile' ? 'active' : ''}`}
              >
                Profile
              </Link>
            </li>
            <li>
              <button
                id="nav-logout"
                aria-label="Logout"
                data-semantic-role="nav-logout"
                onClick={onLogout}
                className="nav-link"
                style={{ background: 'none', border: 'none', cursor: 'pointer' }}
              >
                Logout ({user.username})
              </button>
            </li>
          </>
        ) : (
          <>
            <li>
              <Link
                id="nav-login"
                aria-label="Login"
                data-semantic-role="nav-login"
                to="/login"
                className={`nav-link ${location.pathname === '/login' ? 'active' : ''}`}
              >
                Login
              </Link>
            </li>
            <li>
              <Link
                id="nav-profile"
                aria-label="Profile"
                data-semantic-role="nav-profile"
                to="/profile"
                className={`nav-link ${location.pathname === '/profile' ? 'active' : ''}`}
              >
                Profile
              </Link>
            </li>
          </>
        )}
        <li>
          <button
            id={navCartProps.id}
            aria-label="Cart"
            data-semantic-role="nav-cart"
            onClick={handleCartClick}
            className={navCartProps.className}
            style={{ padding: '0.4rem 0.8rem', fontSize: '0.875rem' }}
          >
            {navCartProps.text || 'Cart'}
            {cartCount > 0 && <span className="cart-count-badge">{cartCount}</span>}
          </button>
        </li>
      </ul>
    </nav>
  );
}

