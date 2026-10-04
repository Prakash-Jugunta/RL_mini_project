import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useMutation } from '../mutations/useMutation.js';

// GROUND-TRUTH METADATA ONLY.
//
// data-semantic-role is used by the environment/evaluator to determine
// the intended UI element and calculate rewards.
//
// It MUST NOT be included in the RL agent's observation/state features.
// Exposing this value to the agent would leak the correct answer.

export default function LoginPage({ onLoginSuccess }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [rememberMe, setRememberMe] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const navigate = useNavigate();

  const { mutate, getDistractors, shouldWrap, shouldReorder } = useMutation();

  const handleSubmit = (e) => {
    e.preventDefault();
    if (username === 'testuser' && password === 'password123') {
      onLoginSuccess({ username });
      navigate('/dashboard');
    } else {
      setErrorMessage('Invalid username or password');
    }
  };

  // Mutated target elements
  const usernameProps = mutate('username-input', {
    id: 'username',
    name: 'username',
    type: 'text',
    placeholder: 'Username',
    label: 'Username',
    className: 'form-input'
  });

  const passwordProps = mutate('password-input', {
    id: 'password',
    name: 'password',
    type: 'password',
    placeholder: 'Password',
    label: 'Password',
    className: 'form-input'
  });

  const loginBtnProps = mutate('login-action', {
    id: 'login-btn',
    type: 'submit',
    text: 'Login',
    className: 'btn btn-primary btn-block'
  });

  const forgotPassProps = mutate('forgot-password-action', {
    id: 'forgot-password',
    text: 'Forgot Password?',
    className: 'forgot-link'
  });

  const rememberMeProps = mutate('remember-me-input', {
    id: 'remember-me',
    name: 'remember',
    type: 'checkbox'
  });

  const distractors = getDistractors('login');
  const isReordered = shouldReorder('login');

  return (
    <div className="card" style={{ maxWidth: '420px', margin: '3rem auto' }}>
      <div className="page-header">
        <h1 className="page-title">Sign In</h1>
        <p className="page-subtitle">Enter your credentials to access your account</p>
      </div>

      {errorMessage && (
        <div style={{ color: '#ef4444', backgroundColor: 'rgba(239, 68, 68, 0.1)', padding: '0.75rem', borderRadius: '0.375rem', marginBottom: '1rem', fontSize: '0.875rem' }}>
          {errorMessage}
        </div>
      )}

      <form onSubmit={handleSubmit}>
        {/* Render Distractor Elements if Level 4/5/6 is active */}
        {distractors.map((d) => (
          <div key={d.id} className="form-group" style={{ marginBottom: '1rem' }}>
            {d.type === 'input' && (
              <>
                <label htmlFor={d.id} className="form-label" style={{ color: '#94a3b8' }}>{d.label}</label>
                <input
                  id={d.id}
                  type={d.inputType || 'text'}
                  className="form-input"
                  placeholder={d.placeholder}
                  data-semantic-role={d.role}
                />
              </>
            )}
            {d.type === 'button' && (
              <button
                id={d.id}
                type="button"
                className="btn btn-secondary btn-block"
                data-semantic-role={d.role}
                onClick={() => alert(`Distractor ${d.text} clicked`)}
              >
                {d.text}
              </button>
            )}
          </div>
        ))}

        {/* Username Field */}
        <div className="form-group">
          <label htmlFor={usernameProps.id} className="form-label">{usernameProps.label || 'Username'}</label>

          {shouldWrap('username-input') ? (
            <div className="dom-wrapper-x1" style={{ border: '1px dashed rgba(255,255,255,0.05)', padding: '0.2rem' }}>
              <section className="dom-inner-x2">
                <input
                  id={usernameProps.id}
                  name={usernameProps.name}
                  type={usernameProps.type}
                  className={usernameProps.className}
                  placeholder={usernameProps.placeholder}
                  aria-label="Username"
                  data-semantic-role="username-input"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  required
                />
              </section>
            </div>
          ) : (
            <input
              id={usernameProps.id}
              name={usernameProps.name}
              type={usernameProps.type}
              className={usernameProps.className}
              placeholder={usernameProps.placeholder}
              aria-label="Username"
              data-semantic-role="username-input"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
          )}
        </div>

        {/* Password Field */}
        <div className="form-group">
          <label htmlFor={passwordProps.id} className="form-label">{passwordProps.label || 'Password'}</label>

          {shouldWrap('password-input') ? (
            <div className="dom-wrapper-x1" style={{ border: '1px dashed rgba(255,255,255,0.05)', padding: '0.2rem' }}>
              <section className="dom-inner-x2">
                <input
                  id={passwordProps.id}
                  name={passwordProps.name}
                  type={passwordProps.type}
                  className={passwordProps.className}
                  placeholder={passwordProps.placeholder}
                  aria-label="Password"
                  data-semantic-role="password-input"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
              </section>
            </div>
          ) : (
            <input
              id={passwordProps.id}
              name={passwordProps.name}
              type={passwordProps.type}
              className={passwordProps.className}
              placeholder={passwordProps.placeholder}
              aria-label="Password"
              data-semantic-role="password-input"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          )}
        </div>

        {/* Realistic non-target distractor elements */}
        <div className="form-group" style={{ flexDirection: isReordered ? 'row-reverse' : 'row', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.5rem' }}>
          <label htmlFor={rememberMeProps.id} className="checkbox-group">
            <input
              id={rememberMeProps.id}
              name={rememberMeProps.name}
              type={rememberMeProps.type}
              aria-label="Remember Me"
              data-semantic-role="remember-me-input"
              checked={rememberMe}
              onChange={(e) => setRememberMe(e.target.checked)}
            />
            <span style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>Remember Me</span>
          </label>

          <a
            id={forgotPassProps.id}
            href="#forgot"
            aria-label="Forgot Password"
            data-semantic-role="forgot-password-action"
            onClick={(e) => { e.preventDefault(); alert('Forgot Password clicked'); }}
            style={{ fontSize: '0.875rem', color: 'var(--primary-color)', textDecoration: 'none' }}
          >
            {forgotPassProps.text || 'Forgot Password?'}
          </a>
        </div>

        {loginBtnProps.renderAs === 'a' ? (
          <a
            id={loginBtnProps.id}
            href="#login"
            role="button"
            aria-label="Login"
            data-semantic-role="login-action"
            onClick={(e) => { e.preventDefault(); handleSubmit(e); }}
            className={loginBtnProps.className}
            style={{ marginTop: '1.5rem', textDecoration: 'none', display: 'inline-block', textAlign: 'center' }}
          >
            {loginBtnProps.text || 'Login'}
          </a>
        ) : (
          <button
            id={loginBtnProps.id}
            aria-label="Login"
            data-semantic-role="login-action"
            type="submit"
            className={loginBtnProps.className}
            style={{ marginTop: '1.5rem' }}
          >
            {loginBtnProps.text || 'Login'}
          </button>
        )}
      </form>
    </div>
  );
}

