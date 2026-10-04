import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { useMutation } from '../mutations/useMutation.js';

/**
 * Development Inspection & Mutation Debug Panel Component
 * Toggled via ?debug=true URL query string or manual toggle button.
 * Displays DOM inspection metadata and active Mutation Level / Seed.
 *
 * CRITICAL NOTE:
 * Developer-only inspection panel. NEVER exposed to RL agent observations.
 */
export default function DebugPanel() {
  const location = useLocation();
  const searchParams = new URLSearchParams(location.search);
  const [isDebugEnabled, setIsDebugEnabled] = useState(searchParams.get('debug') === 'true');
  const [inspectedElement, setInspectedElement] = useState(null);

  const { mutationLevel, seed, levelConfig } = useMutation();

  useEffect(() => {
    if (searchParams.get('debug') === 'true') {
      setIsDebugEnabled(true);
    }
  }, [location.search]);

  useEffect(() => {
    if (!isDebugEnabled) return;

    const handleMouseOver = (e) => {
      const target = e.target;
      if (target.closest('.debug-panel')) return;

      const semanticRole = target.getAttribute('data-semantic-role');
      const ariaLabel = target.getAttribute('aria-label');
      const id = target.id;
      const name = target.getAttribute('name');
      const type = target.getAttribute('type');
      const placeholder = target.getAttribute('placeholder');
      const text = target.innerText?.trim() || target.value || '';

      if (id || semanticRole || ariaLabel || name || target.tagName === 'BUTTON' || target.tagName === 'INPUT' || target.tagName === 'A') {
        setInspectedElement({
          tag: target.tagName,
          id: id || 'N/A',
          name: name || 'N/A',
          type: type || 'N/A',
          text: text ? (text.length > 25 ? text.substring(0, 25) + '...' : text) : 'N/A',
          placeholder: placeholder || 'N/A',
          aria: ariaLabel || 'N/A',
          semanticRole: semanticRole || 'N/A'
        });
      }
    };

    window.addEventListener('mouseover', handleMouseOver);
    return () => window.removeEventListener('mouseover', handleMouseOver);
  }, [isDebugEnabled]);

  if (!isDebugEnabled) {
    return (
      <div style={{ position: 'fixed', bottom: '10px', right: '10px', zIndex: 9999 }}>
        <button
          className="debug-toggle-btn"
          onClick={() => setIsDebugEnabled(true)}
          title="Enable DOM Inspection & Mutation Debug Panel"
        >
          Debug Panel (L{mutationLevel}/S{seed})
        </button>
      </div>
    );
  }

  return (
    <div className="debug-panel" id="debug-inspection-panel">
      <div className="debug-details">
        <div className="debug-item" style={{ borderRight: '1px solid #334155', paddingRight: '1rem' }}>
          <span className="debug-key">Mutation:</span>
          <span className="debug-val" style={{ color: '#f59e0b' }}>
            L{mutationLevel} ({levelConfig?.name?.split('—')[1]?.trim() || 'Original'}) | Seed: {seed}
          </span>
        </div>
        <div className="debug-item">
          <span className="debug-key">Tag:</span>
          <span className="debug-val">{inspectedElement?.tag || '-'}</span>
        </div>
        <div className="debug-item">
          <span className="debug-key">ID:</span>
          <span className="debug-val">{inspectedElement?.id || '-'}</span>
        </div>
        <div className="debug-item">
          <span className="debug-key">Name:</span>
          <span className="debug-val">{inspectedElement?.name || '-'}</span>
        </div>
        <div className="debug-item">
          <span className="debug-key">Type:</span>
          <span className="debug-val">{inspectedElement?.type || '-'}</span>
        </div>
        <div className="debug-item">
          <span className="debug-key">Text:</span>
          <span className="debug-val">{inspectedElement?.text || '-'}</span>
        </div>
        <div className="debug-item">
          <span className="debug-key">Placeholder:</span>
          <span className="debug-val">{inspectedElement?.placeholder || '-'}</span>
        </div>
        <div className="debug-item">
          <span className="debug-key">ARIA:</span>
          <span className="debug-val">{inspectedElement?.aria || '-'}</span>
        </div>
        <div className="debug-item">
          <span className="debug-key">Semantic Role:</span>
          <span className="debug-val" style={{ color: '#a855f7' }}>{inspectedElement?.semanticRole || '-'}</span>
        </div>
      </div>
      <div>
        <button
          className="debug-toggle-btn"
          style={{ backgroundColor: '#ef4444' }}
          onClick={() => setIsDebugEnabled(false)}
        >
          Close
        </button>
      </div>
    </div>
  );
}

