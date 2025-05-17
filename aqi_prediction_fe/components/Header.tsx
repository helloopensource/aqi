import React from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';

const Header: React.FC = () => {
  const router = useRouter();
  
  return (
    <header className="bg-white shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex">
            <div className="flex-shrink-0 flex items-center">
              <Link href="/" legacyBehavior>
                <a className="text-xl font-bold text-blue-600">AQI Prediction</a>
              </Link>
            </div>
            <nav className="hidden sm:ml-6 sm:flex sm:space-x-8">
              <Link href="/" legacyBehavior>
                <a 
                  className={`inline-flex items-center px-1 pt-1 border-b-2 ${
                    router.pathname === '/' 
                      ? 'border-blue-500 text-gray-900' 
                      : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
                  }`}
                >
                  Home
                </a>
              </Link>
              <Link href="/cities" legacyBehavior>
                <a 
                  className={`inline-flex items-center px-1 pt-1 border-b-2 ${
                    router.pathname === '/cities' 
                      ? 'border-blue-500 text-gray-900' 
                      : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
                  }`}
                >
                  Cities
                </a>
              </Link>
              <Link href="/about" legacyBehavior>
                <a 
                  className={`inline-flex items-center px-1 pt-1 border-b-2 ${
                    router.pathname === '/about' 
                      ? 'border-blue-500 text-gray-900' 
                      : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
                  }`}
                >
                  About
                </a>
              </Link>
            </nav>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header; 