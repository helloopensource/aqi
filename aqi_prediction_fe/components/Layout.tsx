import React, { ReactNode } from 'react';
import Head from 'next/head';
import Header from './Header';

interface LayoutProps {
  children: ReactNode;
  title?: string;
}

const Layout: React.FC<LayoutProps> = ({ 
  children, 
  title = 'AQI Prediction' 
}) => {
  return (
    <>
      <Head>
        <title>{title}</title>
        <meta name="description" content="Air Quality Index Prediction for urban areas" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <div className="min-h-screen flex flex-col">
        <Header />
        <main className="flex-grow bg-gray-50">
          <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
            {children}
          </div>
        </main>
        <footer className="bg-white shadow-inner">
          <div className="max-w-7xl mx-auto py-4 px-4 sm:px-6 lg:px-8">
            <p className="text-center text-sm text-gray-500">
              © {new Date().getFullYear()} AQI Prediction System. All rights reserved.
            </p>
          </div>
        </footer>
      </div>
    </>
  );
};

export default Layout; 