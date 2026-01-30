// Basic Flutter widget test for DeepFilter app.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:deepfilter_test/main.dart';

void main() {
  testWidgets('DeepFilter app smoke test', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const DeepFilterApp());

    // Verify that the app title is displayed
    expect(find.text('DeepFilter Test'), findsOneWidget);
  });
}
