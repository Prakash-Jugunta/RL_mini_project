import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './components/Navbar';
import DebugPanel from './components/DebugPanel';
import LoginPage from './pages/LoginPage';
import DashboardPage from './pages/DashboardPage';
import ProductCatalogPage from './pages/ProductCatalogPage';
import ProductDetailPage from './pages/ProductDetailPage';
import ProfilePage from './pages/ProfilePage';
import CartPage from './pages/CartPage';
import CheckoutPage from './pages/CheckoutPage';
import OrderSuccessPage from './pages/OrderSuccessPage';
import { PRODUCTS } from './data/seedData';
import { MutationProvider } from './mutations/mutationContext';

// GROUND-TRUTH METADATA ONLY.
//
// data-semantic-role is used by the environment/evaluator to determine
// the intended UI element and calculate rewards.
//
// It MUST NOT be included in the RL agent's observation/state features.
// Exposing this value to the agent would leak the correct answer.

export default function App() {
  const [user, setUser] = useState(null);
  const [cartItems, setCartItems] = useState([]);

  const handleLoginSuccess = (userData) => {
    setUser({
      ...userData,
      name: userData.name || 'Test User',
      email: userData.email || 'testuser@example.com'
    });
  };

  const handleLogout = () => {
    setUser(null);
  };

  const handleAddToCart = (product) => {
    setCartItems((prevItems) => {
      const existing = prevItems.find((item) => item.id === product.id);
      if (existing) {
        return prevItems.map((item) =>
          item.id === product.id ? { ...item, quantity: (item.quantity || 1) + 1 } : item
        );
      }
      return [...prevItems, { ...product, quantity: 1 }];
    });
  };

  const handleRemoveFromCart = (productId) => {
    setCartItems((prevItems) => prevItems.filter((item) => item.id !== productId));
  };

  const handleClearCart = () => {
    setCartItems([]);
  };

  const handleUpdateProfile = (updatedProfile) => {
    setUser((prev) => ({
      ...prev,
      ...updatedProfile
    }));
  };

  return (
    <Router>
      <MutationProvider>
        <div className="app-container">
          <Navbar
            cartCount={cartItems.reduce((sum, item) => sum + (item.quantity || 1), 0)}
            user={user}
            onLogout={handleLogout}
          />
          <main className="main-content">
            <Routes>
              <Route path="/" element={<Navigate to="/products" replace />} />
              <Route
                path="/login"
                element={<LoginPage onLoginSuccess={handleLoginSuccess} />}
              />
              <Route
                path="/dashboard"
                element={<DashboardPage user={user} />}
              />
              <Route
                path="/products"
                element={<ProductCatalogPage onAddToCart={handleAddToCart} />}
              />
              <Route
                path="/products/:productId"
                element={<ProductDetailPage onAddToCart={handleAddToCart} />}
              />
              <Route
                path="/profile"
                element={<ProfilePage user={user} onUpdateProfile={handleUpdateProfile} />}
              />
              <Route
                path="/cart"
                element={
                  <CartPage
                    cartItems={cartItems}
                    onRemoveFromCart={handleRemoveFromCart}
                    onClearCart={handleClearCart}
                  />
                }
              />
              <Route
                path="/checkout"
                element={
                  <CheckoutPage
                    cartItems={cartItems}
                    onCompleteOrder={handleClearCart}
                  />
                }
              />
              <Route
                path="/order-success"
                element={<OrderSuccessPage />}
              />
              <Route path="*" element={<Navigate to="/products" replace />} />
            </Routes>
          </main>
          <DebugPanel />
        </div>
      </MutationProvider>
    </Router>
  );
}

