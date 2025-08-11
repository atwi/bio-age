import React from 'react';
import { View, StyleSheet } from 'react-native';
import { Spinner, Text } from '@ui-kitten/components';

const LoadingSpinner = ({ message = "Loading..." }) => {
  return (
    <View style={styles.container}>
      <Spinner size='large' />
      <Text style={styles.message}>{message}</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#0E1116',
    minHeight: 200,
  },
  message: {
    marginTop: 16,
    fontSize: 16,
    color: '#E6EAF2',
    textAlign: 'center',
  }
});

export default LoadingSpinner; 