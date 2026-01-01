import React from 'react';
import styles from './LoadingDots.module.css';

export interface LoadingDotsProps {
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export const LoadingDots: React.FC<LoadingDotsProps> = ({ size = 'md', className = '' }) => {
  const dotsClasses = [styles.loadingDots, styles[size], className].filter(Boolean).join(' ');

  return (
    <div className={dotsClasses} role="status" aria-label="Loading">
      <span className={styles.dot} />
      <span className={styles.dot} />
      <span className={styles.dot} />
      <span className={styles.srOnly}>Loading...</span>
    </div>
  );
};

export default LoadingDots;
