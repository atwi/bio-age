import React from 'react';
import { Platform, View } from 'react-native';

export default function GlassPanel({ style, children, rounded = 16, borderOpacity = 0.1, backgroundOpacity = 0.55, blurPx = 10 }) {
  const common = {
    borderRadius: rounded,
    borderWidth: 1,
    borderColor: `rgba(255,255,255,${borderOpacity})`,
    overflow: 'hidden',
  };

  const webStyle = Platform.OS === 'web' ? {
    backdropFilter: `blur(${blurPx}px)`,
    WebkitBackdropFilter: `blur(${blurPx}px)`,
  } : {};

  const fallbackBg = {
    backgroundColor: `rgba(28,34,44,${backgroundOpacity})`,
  };

  return (
    <View style={[common, fallbackBg, webStyle, style]}>
      {children}
    </View>
  );
} 