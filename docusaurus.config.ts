import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'A Comprehensive Guide to Embodied AI Systems',
  favicon: 'img/favicon.ico',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: 'https://hackathone-q4-ebook.vercel.app/',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'Ibrahimsabir', // Usually your GitHub org/user name.
  projectName: 'hackathoneQ4-ebook', // Usually your repo name.

  onBrokenLinks: 'throw',

  // Custom fields for runtime configuration
  customFields: {
    // Backend API URL - will be set via environment variable in Vercel
    BACKEND_URL: process.env.BACKEND_URL || 'http://localhost:8000',
  },

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    image: 'img/book-social-card.jpg',
    colorMode: {
      respectPrefersColorScheme: true,
    },
    stylesheets: [
      'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap',
    ],
    navbar: {
      title: 'Physical AI & Humanoid Robotics',
      logo: {
        alt: 'Physical AI & Humanoid Robotics Logo',
        src: 'img/book-logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Book Chapters',
        },
        {
          to: '/chatbot',
          label: '🤖 AI Chatbot',
          position: 'left',
        },
        {
          href: 'https://github.com/Ibrahimsabir/hackathoneQ4-ebook',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Book Content',
          items: [
            {
              label: 'Introduction',
              to: '/docs/intro',
            },
            {
              label: 'All Chapters',
              to: '/docs/intro',
            },
          ],
        },
        {
          title: 'Topics',
          items: [
            {
              label: 'Physical AI',
              to: '/docs/intro',
            },
            {
              label: 'Humanoid Robotics',
              to: '/docs/intro',
            },
            {
              label: 'ROS 2 Systems',
              to: '/docs/intro',
            },
          ],
        },
        {
          title: 'Resources',
          items: [
            {
              label: 'GitHub Repository',
              href: 'https://github.com/Ibrahimsabir/hackathoneQ4-ebook',
            },
            {
              label: 'Live Demo',
              href: 'https://hackathone-q4-ebook.vercel.app/',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Book. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
