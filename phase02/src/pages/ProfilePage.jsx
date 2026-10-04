import React, { useState } from 'react';
import { useMutation } from '../mutations/useMutation.js';

// GROUND-TRUTH METADATA ONLY.
//
// data-semantic-role is used by the environment/evaluator to determine
// the intended UI element and calculate rewards.
//
// It MUST NOT be included in the RL agent's observation/state features.
// Exposing this value to the agent would leak the correct answer.

export default function ProfilePage({ user, onUpdateProfile }) {
  const [name, setName] = useState(user?.name || 'Test User');
  const [email, setEmail] = useState(user?.email || 'testuser@example.com');
  const [success, setSuccess] = useState(false);

  const { mutate, getDistractors, shouldWrap } = useMutation();

  const handleSubmit = (e) => {
    e.preventDefault();
    if (onUpdateProfile) {
      onUpdateProfile({ name, email });
    }
    setSuccess(true);
  };

  const nameProps = mutate('profile-name-input', {
    id: 'profile-name',
    name: 'name',
    type: 'text',
    placeholder: 'Full Name',
    label: 'Full Name',
    className: 'form-input'
  });

  const emailProps = mutate('profile-email-input', {
    id: 'profile-email',
    name: 'email',
    type: 'email',
    placeholder: 'Email',
    label: 'Email Address',
    className: 'form-input'
  });

  const saveBtnProps = mutate('profile-save-action', {
    id: 'save-profile-btn',
    type: 'submit',
    text: 'Save Profile',
    className: 'btn btn-primary btn-block'
  });

  const distractors = getDistractors('profile');

  return (
    <div className="card" style={{ maxWidth: '480px', margin: '2rem auto' }}>
      <div className="page-header">
        <h1 className="page-title">User Profile</h1>
        <p className="page-subtitle">Manage account profile and communication settings</p>
      </div>

      <form onSubmit={handleSubmit}>
        {distractors.map((d) => (
          <div key={d.id} className="form-group" style={{ marginBottom: '1rem' }}>
            <label htmlFor={d.id} className="form-label" style={{ color: '#94a3b8' }}>{d.label}</label>
            {d.type === 'input' && (
              <input
                id={d.id}
                type={d.inputType || 'text'}
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

        <div className="form-group">
          <label htmlFor={nameProps.id} className="form-label">{nameProps.label || 'Full Name'}</label>
          {shouldWrap('profile-name-input') ? (
            <div className="dom-wrapper-x1">
              <input
                id={nameProps.id}
                name={nameProps.name}
                type={nameProps.type}
                className={nameProps.className}
                placeholder={nameProps.placeholder}
                aria-label="Full Name"
                data-semantic-role="profile-name-input"
                value={name}
                onChange={(e) => {
                  setName(e.target.value);
                  setSuccess(false);
                }}
                required
              />
            </div>
          ) : (
            <input
              id={nameProps.id}
              name={nameProps.name}
              type={nameProps.type}
              className={nameProps.className}
              placeholder={nameProps.placeholder}
              aria-label="Full Name"
              data-semantic-role="profile-name-input"
              value={name}
              onChange={(e) => {
                setName(e.target.value);
                setSuccess(false);
              }}
              required
            />
          )}
        </div>

        <div className="form-group">
          <label htmlFor={emailProps.id} className="form-label">{emailProps.label || 'Email Address'}</label>
          {shouldWrap('profile-email-input') ? (
            <div className="dom-wrapper-x1">
              <input
                id={emailProps.id}
                name={emailProps.name}
                type={emailProps.type}
                className={emailProps.className}
                placeholder={emailProps.placeholder}
                aria-label="Email"
                data-semantic-role="profile-email-input"
                value={email}
                onChange={(e) => {
                  setEmail(e.target.value);
                  setSuccess(false);
                }}
                required
              />
            </div>
          ) : (
            <input
              id={emailProps.id}
              name={emailProps.name}
              type={emailProps.type}
              className={emailProps.className}
              placeholder={emailProps.placeholder}
              aria-label="Email"
              data-semantic-role="profile-email-input"
              value={email}
              onChange={(e) => {
                setEmail(e.target.value);
                setSuccess(false);
              }}
              required
            />
          )}
        </div>

        <button
          id={saveBtnProps.id}
          aria-label="Save Profile"
          data-semantic-role="profile-save-action"
          type="submit"
          className={saveBtnProps.className}
          style={{ marginTop: '1.5rem' }}
        >
          {saveBtnProps.text || 'Save Profile'}
        </button>
      </form>

      {success && (
        <div
          id="profile-success"
          className="alert-success"
          data-semantic-role="profile-success-message"
        >
          Profile updated successfully
        </div>
      )}
    </div>
  );
}

