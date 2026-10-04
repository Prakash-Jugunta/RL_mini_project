import React from 'react';
import { Link } from 'react-router-dom';

export default function DashboardPage({ user }) {
  const username = user?.username || 'testuser';

  return (
    <div className="card">
      <div className="page-header">
        <h1 className="page-title" data-semantic-role="dashboard-welcome">Welcome, {username}</h1>
        <p className="page-subtitle">User Dashboard & Portal Overview</p>
      </div>

      <div style={{ margin: '2rem 0', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
        <Link to="/products" className="card" style={{ textDecoration: 'none', color: 'inherit', textAlign: 'center' }}>
          <h3 style={{ color: 'var(--primary-color)', marginBottom: '0.5rem' }}>Browse Catalog</h3>
          <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>Explore products and search inventory.</p>
        </Link>
        <Link to="/profile" className="card" style={{ textDecoration: 'none', color: 'inherit', textAlign: 'center' }}>
          <h3 style={{ color: 'var(--primary-color)', marginBottom: '0.5rem' }}>Edit Profile</h3>
          <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>Update your name and email address.</p>
        </Link>
        <Link to="/cart" className="card" style={{ textDecoration: 'none', color: 'inherit', textAlign: 'center' }}>
          <h3 style={{ color: 'var(--primary-color)', marginBottom: '0.5rem' }}>View Shopping Cart</h3>
          <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>Proceed to order checkout.</p>
        </Link>
      </div>
    </div>
  );
}

