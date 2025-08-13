import React, { useRef, useState } from 'react';
import { Animated, Easing, Platform, TouchableOpacity, View } from 'react-native';
import { Layout, Text, Icon } from '@ui-kitten/components';
import GlassPanel from './GlassPanel';

const AccordionItem = ({ title, children, initiallyOpen = false }) => {
  const [open, setOpen] = useState(initiallyOpen);
  const contentHeight = useRef(new Animated.Value(initiallyOpen ? 1 : 0)).current;
  const contentOpacity = useRef(new Animated.Value(initiallyOpen ? 1 : 0)).current;
  const rotate = contentHeight.interpolate({ inputRange: [0, 1], outputRange: ['0deg', '180deg'] });

  const toggle = () => {
    const to = open ? 0 : 1;
    setOpen(!open);
    Animated.parallel([
      Animated.timing(contentHeight, { toValue: to, duration: 180, easing: Easing.out(Easing.cubic), useNativeDriver: false }),
      Animated.timing(contentOpacity, { toValue: to, duration: 160, easing: Easing.out(Easing.cubic), useNativeDriver: true }),
    ]).start();
  };

  return (
    <GlassPanel style={{ borderRadius: 16, padding: 12, marginBottom: 10, borderColor: 'rgba(255,255,255,0.06)' }}>
      <TouchableOpacity onPress={toggle} activeOpacity={0.85} style={{ flexDirection: 'row', alignItems: 'center' }}>
        <Text style={{ flex: 1, color: '#E6EAF2', fontWeight: '700', fontSize: 14 }}>{title}</Text>
        <Animated.View style={{ transform: [{ rotate }] }}>
          <Icon name={open ? 'chevron-up-outline' : 'chevron-down-outline'} fill="#9AA3AF" style={{ width: 18, height: 18 }} />
        </Animated.View>
      </TouchableOpacity>
      <Animated.View style={{ overflow: 'hidden', opacity: contentOpacity, maxHeight: contentHeight.interpolate({ inputRange: [0, 1], outputRange: [0, 400] }) }}>
        <View style={{ height: 8 }} />
        <Text style={{ color: '#9AA3AF', fontSize: 13, lineHeight: 18 }}>{children}</Text>
      </Animated.View>
    </GlassPanel>
  );
};

export default function FAQAccordion() {
  return (
    <Layout style={{ backgroundColor: 'transparent', width: '100%', maxWidth: 500, alignSelf: 'center' }}>
      <Text style={{ fontSize: 13, fontWeight: '700', color: '#E6EAF2', letterSpacing: 0.2, marginBottom: 10 }}>FAQ</Text>
      <AccordionItem title="What is TrueAge?" initiallyOpen={false}>
        TrueAge estimates how old you look using research‑grade AI models including Harvard FaceAge, DeepFace and GPT‑Vision. It provides both biological and perceived age estimates.
      </AccordionItem>
      <AccordionItem title="How does TrueAge work?">
        Your photo is processed by face detection and multiple age estimation models. We combine their outputs and return an easy‑to‑read result with optional insights about factors that influence perceived age.
      </AccordionItem>
      <AccordionItem title="Is my photo stored?">
        No permanent storage. Images are processed temporarily to generate predictions and then discarded. See Privacy Policy for details.
      </AccordionItem>
      <AccordionItem title="How do I improve the accuracy?">
        Use a clear, front‑facing photo in good, even lighting. Avoid sunglasses, extreme makeup, or heavy filters. Make sure your face fills most of the frame and is centered.
      </AccordionItem>
    </Layout>
  );
} 