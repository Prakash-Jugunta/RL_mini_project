import { useContext } from 'react';
import { MutationContext } from './mutationContext';

/**
 * Custom hook to consume mutation context.
 * Split into a separate file to avoid Vite Fast Refresh HMR incompatibility.
 */
export function useMutation() {
  return useContext(MutationContext);
}
