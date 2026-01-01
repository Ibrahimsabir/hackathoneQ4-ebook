import React from 'react';
import styles from './Spinner.module.css';

export type SpinnerSize = 'sm' | 'md' | 'lg';

export interface SpinnerProps {
  size?: SpinnerSize;
  className?: string;
}

export const Spinner: React.FC<SpinnerProps> = ({ size = 'md', className = '' }) => {
  const spinnerClasses = [styles.spinner, styles[size], className].filter(Boolean).join(' ');

  return (
    <div className={spinnerClasses} role="status" aria-label="Loading">
      <svg className={styles.svg} viewBox="0 0 50 50">
        <circle
          className={styles.circle}
          cx="25"
          cy="25"
          r="20"
          fill="none"
          strokeWidth="4"
        />
      </svg>
      <span className={styles.srOnly}>Loading...</span>
    </div>
  );
};

export default Spinner;
