import React, { useEffect, useRef, useState } from 'react';
import { Animated, Easing, TouchableOpacity, View } from 'react-native';
import { Layout, Text, Icon, Divider } from '@ui-kitten/components';
import GlassPanel from './GlassPanel';

const AccordionItem = ({ title, children, initiallyOpen = false }) => {
  const [open, setOpen] = useState(initiallyOpen);
  const [measuredHeight, setMeasuredHeight] = useState(0);
  const heightAnim = useRef(new Animated.Value(initiallyOpen ? 1 : 0)).current;
  const opacityAnim = useRef(new Animated.Value(initiallyOpen ? 1 : 0)).current;

  useEffect(() => {
    // If initially open and we get a height later, sync the animation value to open
    if (initiallyOpen && measuredHeight > 0) {
      heightAnim.setValue(1);
      opacityAnim.setValue(1);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [measuredHeight]);

  const toggle = () => {
    const to = open ? 0 : 1;
    setOpen(!open);
    Animated.parallel([
      Animated.timing(heightAnim, { toValue: to, duration: 200, easing: Easing.out(Easing.cubic), useNativeDriver: false }),
      Animated.timing(opacityAnim, { toValue: to, duration: 180, easing: Easing.out(Easing.cubic), useNativeDriver: true }),
    ]).start();
  };

  const animatedHeight = heightAnim.interpolate({ inputRange: [0, 1], outputRange: [0, Math.max(measuredHeight, 0)] });

  return (
    <View>
      <TouchableOpacity
        onPress={toggle}
        activeOpacity={0.85}
        accessibilityRole="button"
        accessibilityState={{ expanded: open }}
        style={{ flexDirection: 'row', alignItems: 'center', paddingVertical: 10 }}
      >
        <Text style={{ flex: 1, color: '#E6EAF2', fontWeight: '700', fontSize: 14 }}>{title}</Text>
        <Animated.View style={{ transform: [{ rotate: heightAnim.interpolate({ inputRange: [0, 1], outputRange: ['0deg', '180deg'] }) }] }}>
          <Icon name="chevron-down-outline" fill="#9AA3AF" style={{ width: 18, height: 18 }} />
        </Animated.View>
      </TouchableOpacity>

      <Animated.View style={{ overflow: 'hidden', height: animatedHeight, opacity: opacityAnim }}>
        <View
          style={{ paddingTop: 4, paddingBottom: 8 }}
          onLayout={(e) => setMeasuredHeight(e.nativeEvent.layout.height)}
        >
          <Text style={{ color: '#9AA3AF', fontSize: 13, lineHeight: 18 }}>{children}</Text>
        </View>
      </Animated.View>
    </View>
  );
};

export default function FAQAccordion() {
  const items = [
    {
      title: 'What is TrueAge?',
      body: 'TrueAge estimates how old you look using research‑grade AI models including Harvard FaceAge, DeepFace and GPT‑Vision. It provides both biological and perceived age estimates.',
    },
    {
      title: 'How does TrueAge work?',
      body: 'Your photo is processed by face detection and multiple age estimation models. We combine their outputs and return an easy‑to‑read result with optional insights about factors that influence perceived age.',
    },
    {
      title: 'Is my photo stored?',
      body: 'No permanent storage. Images are processed temporarily to generate predictions and then discarded. See Privacy Policy for details.',
    },
    {
      title: 'How do I improve the accuracy?',
      body: 'Use a clear, front‑facing photo in good, even lighting. Avoid sunglasses, extreme makeup, or heavy filters. Make sure your face fills most of the frame and is centered.',
    },
  ];

  return (
    <Layout style={{ backgroundColor: 'transparent', width: '100%', alignSelf: 'center' }}>
      {/* Heading outside the card */}
      <View style={{ width: '100%', maxWidth: 500, alignSelf: 'center', marginBottom: 8 }}>
        <View style={{ flexDirection: 'row', alignItems: 'center' }}>
          <View style={{ flexDirection: 'row', alignItems: 'center', borderRadius: 999, paddingHorizontal: 10, paddingVertical: 4 }}>
            <Text category='s1' style={{ color: '#E6EAF2' }}>FAQs</Text>
          </View>
        </View>
      </View>

      {/* Card with items only */}
      <GlassPanel style={{ borderRadius: 16, padding: 14, borderColor: 'rgba(255,255,255,0.06)', width: '100%', alignSelf: 'center' }}>
        {items.map((it, idx) => (
          <View key={it.title}>
            <AccordionItem title={it.title}>
              {it.body}
            </AccordionItem>
            {idx < items.length - 1 && <Divider style={{ backgroundColor: 'rgba(27,32,43,0.8)' }} />}
          </View>
        ))}
      </GlassPanel>
    </Layout>
  );
} 