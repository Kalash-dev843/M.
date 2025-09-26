import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import * as Localization from 'expo-localize';

import en from './translations/en.json';
import hi from './translations/hi.json';

i18n
  .use(initReactI18next)
  .init({
    compatibilityJSON: 'v3',
    lng: Localization.locale.startsWith('hi') ? 'hi' : 'en',
    fallbackLng: 'en',
    resources: {
      en: { translation: en },
      hi: { translation: hi },
    },
    interpolation: { escapeValue: false },
  });

export default i18n;
