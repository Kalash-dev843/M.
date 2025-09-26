"use client";

import 'react-native-gesture-handler';
import { AppRegistry } from 'react-native';
import React, { useState, useEffect, useCallback } from "react";
import { NavigationContainer } from "@react-navigation/native";
import { createStackNavigator } from "@react-navigation/stack";
import { Alert, Image, Text, TextInput, View, TouchableOpacity, Modal, StyleSheet, FlatList, ActivityIndicator, StatusBar, Platform } from "react-native";
import { SafeAreaView, SafeAreaProvider } from 'react-native-safe-area-context';
import AppLogo from "./assets/images/app-logo.jpg";
import * as ImagePicker from "expo-image-picker";
import { useFocusEffect } from "@react-navigation/native";
import * as Location from 'expo-location';
import * as FileSystem from 'expo-file-system';
import ProjectSelectionScreen from "./ProjectSelectionScreen"; 
import { 
  LanguageProvider, 
  LanguageToggle, 
  LanguageSelector, 
  useLanguage 
} from "./LanguageContext";


// Server URL (update to your FastAPI Render backend)
const SERVER_URL = "https://aquachain.onrender.com";
//const SERVER_URL = "http://10.166.74.232:8000";
const ENDPOINT_PATH = "/api/reports";


// ===================== Custom Button =====================
const CustomButton = ({ title, onPress }) => (
  <TouchableOpacity style={styles.button} onPress={onPress} activeOpacity={0.8}>
    <Text style={styles.buttonText}>{title}</Text>
  </TouchableOpacity>
);

// ===================== Screen 1: Login =====================
function LoginScreen({ navigation }) {
  const [agentName, setAgentName] = React.useState("");
  const [agentId, setAgentId] = React.useState("");
  const [agentPass, setAgentPass] = React.useState("");
  const [isExistingAgent, setIsExistingAgent] = React.useState(true);

  
  const handleContinue = async () => {
    if (!agentId.trim() || !agentPass.trim() || (!isExistingAgent && !agentName.trim())) {
      alert("Please fill in all fields.");
      return;
    }

    try {
      if (isExistingAgent) {
        // LOGIN
        const response = await fetch(`${SERVER_URL}/api/login`, {
          method: "POST",
          headers: {
            "Content-Type": "application/x-www-form-urlencoded",
          },
          body: new URLSearchParams({
            email: agentId.trim(),
            password: agentPass,
          }).toString(),
        });

        const data = await response.json();
        if (data.success) {
          alert("Login successful!");
          // Navigate to ProjectSelection instead of Main
          navigation.replace("ProjectSelection", { 
            name: data.username || agentName.trim(), 
            agentId 
          });
        } else {
          alert(data.message || "Invalid credentials");
        }
      } else {
        // SIGNUP
        const response = await fetch(`${SERVER_URL}/api/register`, {
          method: "POST",
          headers: {
            "Content-Type": "application/x-www-form-urlencoded",
          },
          body: new URLSearchParams({
            email: agentId.trim(),
            username: agentName.trim(),
            password: agentPass,
          }).toString(),
        });

        const data = await response.json();
        if (data.success) {
          alert("Account created successfully!");
          // Navigate to ProjectSelection instead of Main
          navigation.replace("ProjectSelection", { 
            name: agentName.trim(), 
            agentId 
          });
        } else {
          alert(data.message || "Signup failed");
        }
      }
    } catch (err) {
      console.error(err);
      alert("Something went wrong. Please try again.");
    }
  };

  return (
    <SafeAreaView style={styles.container} edges={["right", "bottom", "left"]}>
      <StatusBar barStyle="light-content" backgroundColor="#1e40af" />
      <View style={styles.loginCard}>
        <Image source={AppLogo} style={styles.bigLogo} />
        <Text style={styles.title}>AquaChain</Text>
        <Text style={styles.subtitle}>Field Data Collection Portal</Text>

        {/* Toggle Login / Signup */}
        <View style={styles.toggleContainer}>
          <TouchableOpacity
            style={[styles.toggleButton, isExistingAgent && styles.toggleActive]}
            onPress={() => setIsExistingAgent(true)}
          >
            <Text style={[styles.toggleText, isExistingAgent && styles.toggleActiveText]}>Login</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.toggleButton, !isExistingAgent && styles.toggleActive]}
            onPress={() => setIsExistingAgent(false)}
          >
            <Text style={[styles.toggleText, !isExistingAgent && styles.toggleActiveText]}>Signup</Text>
          </TouchableOpacity>
        </View>

        {/* Input Fields */}
        {!isExistingAgent && (
          <TextInput
            style={styles.input}
            placeholder="Enter your name"
            placeholderTextColor="#6B7280"
            value={agentName}
            onChangeText={setAgentName}
          />
        )}

        <TextInput
          style={styles.input}
          placeholder="Enter Agent ID or Email"
          placeholderTextColor="#6B7280"
          value={agentId}
          onChangeText={setAgentId}
          keyboardType="email-address"
          autoCapitalize="none"
        />
        <TextInput
          style={styles.input}
          placeholder="Enter Password"
          placeholderTextColor="#6B7280"
          value={agentPass}
          onChangeText={setAgentPass}
          secureTextEntry
        />

        <CustomButton title={isExistingAgent ? "Login" : "Signup"} onPress={handleContinue} />
      </View>
    </SafeAreaView>
  );
}

// ===================== Screen 2: Main =====================
function MainScreen({ route, navigation }) {
  const { name, agentId, project, projectName, projectId } = route.params || {};
  const { t } = useLanguage(); 
  const [langModalVisible, setLangModalVisible] = useState(false);

  const handleLogout = () => {
    navigation.replace("Login");
  };

  const handleBackToProjects = () => {
    navigation.navigate("ProjectSelection", { name, agentId });
  };

  return (
    <SafeAreaView style={styles.container} edges={['right', 'bottom', 'left']}>
      <StatusBar barStyle="light-content" backgroundColor="#1e40af" />

      {/* Header */}
      <View style={styles.headerWrap}>
        <Image source={AppLogo} style={styles.smallLogo} />
        <Text style={styles.title}>
          {t("hello")}, {name ? name : t("untitled")} 👋
        </Text>
        <Text style={styles.subtitle}>
          {t("agentId")}: {agentId ? agentId : t("na")}
        </Text>

        {/* Show selected project */}
        <View style={styles.projectDisplay}>
          <Text style={styles.projectLabel}>Selected Project:</Text>
          <Text style={styles.projectValue}>{project || projectName || "Not selected"}</Text>
        </View>

        <Text style={styles.subtitle}>
          {t("whatToDo")}
        </Text>

        {/* Back to Projects button */}
        <TouchableOpacity style={styles.backButton} onPress={handleBackToProjects}>
          <Text style={styles.backButtonText}>← Projects</Text>
        </TouchableOpacity>

        {/* Logout button in corner */}
        <TouchableOpacity style={styles.logoutButton} onPress={handleLogout}>
          <Text style={styles.logoutText}>{t("logout")}</Text>
        </TouchableOpacity>

        {/* Language toggle in corner */}
        <LanguageToggle onPress={() => setLangModalVisible(true)} />
        <LanguageSelector 
          visible={langModalVisible} 
          onClose={() => setLangModalVisible(false)} 
        />
      </View>

      {/* Main Buttons */}
      <View style={{ width: "100%" }}>
        <CustomButton
          title={t("createReport")}
          onPress={() => navigation.navigate("Report", { 
            name, 
            agentId, 
            projectName: project || projectName,
            projectId 
          })}
        />

        <CustomButton
          title={t("viewReports")}
          onPress={() => navigation.navigate("SavedReports", { 
            name, 
            agentId, 
            projectName: project || projectName,
            projectId 
          })}
        />
      </View>
    </SafeAreaView>
  );
}

// ===================== Screen 3: Report =====================
function ReportScreen({ route, navigation }) {
  const { name, agentId, projectName, projectId } = route.params || {};
  const [localProjectName, setLocalProjectName] = useState(projectName || "");
  const [photos, setPhotos] = useState([]);
  const [gps, setGps] = useState("");
  const [modalVisible, setModalVisible] = useState(false);
  const [selectedPhoto, setSelectedPhoto] = useState(null);

  useEffect(() => {
    (async () => {
      let { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission denied', 'Location permission is required for geotagging.');
      }
    })();
  }, []);

  const takePhoto = async () => {
    let result = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      aspect: [4, 3],
      quality: 0.7,
    });

    if (!result.canceled && result.assets && result.assets.length > 0) {
      const uri = result.assets[0].uri;
      const datetime = new Date().toISOString();
      let gps_latitude = null;
      let gps_longitude = null;
      let file_size = 0;

      try {
        let location = await Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.High });
        gps_latitude = location.coords.latitude;
        gps_longitude = location.coords.longitude;
      } catch (error) {
        console.log('Location error:', error);
        Alert.alert('Location Error', 'Could not get GPS location.');
      }

      try {
        const stat = await FileSystem.getInfoAsync(uri);
        file_size = stat.size || 0;
      } catch (error) {
        console.log('File size error:', error);
      }

      setPhotos([...photos, { uri, datetime, gps_latitude, gps_longitude, file_size }]);
    }
  };

  const getLocation = async () => {
    try {
      let location = await Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.High });
      const gpsString = `${location.coords.latitude.toFixed(4)}° N, ${location.coords.longitude.toFixed(4)}° E`;
      setGps(gpsString);
    } catch (error) {
      console.log('Location error:', error);
      Alert.alert('Location Error', 'Could not get GPS location.');
    }
  };

  const handleSubmit = async () => {
    const currentProjectName = localProjectName || projectName;
    if (!currentProjectName) {
      Alert.alert("Validation", "Please enter Project Name.");
      return;
    }
    if (photos.length < 3) {
      Alert.alert("Validation", "Please take at least 3 photos.");
      return;
    }

    const formData = new FormData();
    formData.append('projectName', currentProjectName);
    formData.append('gps', gps);
    formData.append('submittedAt', new Date().toISOString());
    formData.append('agentId', agentId || '');
    formData.append('agentName', name || '');

    photos.forEach((photo, index) => {
      formData.append('photos', {
        uri: photo.uri,
        name: `photo_${index + 1}.jpg`,
        type: 'image/jpeg',
      });
    });

    try {
      const res = await fetch(`${SERVER_URL}${ENDPOINT_PATH}`, {
        method: "POST",
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (!res.ok) throw new Error(`Server error: ${res.status}`);

      const body = await res.json();
      Alert.alert("Success!", `Report uploaded. Server id: ${body.id || "unknown"}`);
      clearForm();
      navigation.goBack();
    } catch (err) {
      console.log("POST error:", err);
      Alert.alert("Upload failed", "Could not reach backend. Please try again.");
    }
  };

  const clearForm = () => {
    setLocalProjectName(projectName || "");
    setPhotos([]);
    setGps("");
  };

  useFocusEffect(
    useCallback(() => {
      return () => clearForm();
    }, [])
  );

  return (
    <SafeAreaView style={styles.container} edges={['right', 'bottom', 'left']}>
      <StatusBar barStyle="light-content" backgroundColor="#1e40af" />
      
      <Text style={styles.label}>Project Name:</Text>
      <TextInput
        style={styles.input}
        placeholder="Enter project name"
        placeholderTextColor="#6B7280"
        value={localProjectName}
        onChangeText={setLocalProjectName}
        editable={!projectName} // If projectName is provided, make it read-only
      />
      
      <View style={{ marginVertical: 8, width: "100%" }}>
        <CustomButton title={`Add Photo (${photos.length}/3+)`} onPress={takePhoto} />
        {photos.length > 0 && (
          <FlatList
            data={photos}
            keyExtractor={(item, idx) => item.uri + String(idx)}
            horizontal
            showsHorizontalScrollIndicator={false}
            contentContainerStyle={{ gap: 10, marginTop: 12 }}
            renderItem={({ item }) => (
              <TouchableOpacity onPress={() => {
                setSelectedPhoto(item);
                setModalVisible(true);
              }}>
                <Image source={{ uri: item.uri }} style={styles.preview} />
              </TouchableOpacity>
            )}
          />
        )}
      </View>
      
      <View style={{ marginVertical: 8, width: "100%" }}>
        <CustomButton title="Get GPS Location" onPress={getLocation} />
        {gps && <Text style={styles.gps}>📍 {String(gps)}</Text>}
      </View>
      
      <View style={{ marginTop: 10, width: "100%" }}>
        <CustomButton title="Submit" onPress={handleSubmit} />
      </View>
      
      <Modal
        animationType="slide"
        transparent={true}
        visible={modalVisible}
        onRequestClose={() => setModalVisible(false)}
      >
        <View style={styles.modalView}>
          {selectedPhoto && (
            <>
              <Image source={{ uri: selectedPhoto.uri }} style={styles.modalImage} />
              <Text style={styles.modalText}>Datetime: {String(selectedPhoto.datetime)}</Text>
              <Text style={styles.modalText}>GPS Latitude: {String(selectedPhoto.gps_latitude?.toFixed(4) || 'N/A')}</Text>
              <Text style={styles.modalText}>GPS Longitude: {String(selectedPhoto.gps_longitude?.toFixed(4) || 'N/A')}</Text>
              <Text style={styles.modalText}>File Size: {String(selectedPhoto.file_size ? `${selectedPhoto.file_size} bytes` : 'N/A')}</Text>
              <TouchableOpacity
                style={styles.button}
                onPress={() => setModalVisible(false)}
              >
                <Text style={styles.buttonText}>Close</Text>
              </TouchableOpacity>
            </>
          )}
        </View>
      </Modal>
    </SafeAreaView>
  );
}

// ===================== Screen 4: Saved Reports =====================
function SavedReportsScreen({ route }) {
  const { projectName, projectId } = route.params || {};
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [modalVisible, setModalVisible] = useState(false);
  const [selectedPhoto, setSelectedPhoto] = useState(null);

  const fetchReports = async () => {
    try {
      setLoading(true);
      setError("");
      let url = `${SERVER_URL}${ENDPOINT_PATH}`;
      
      // If we have a specific project, we could filter by it
      if (projectName) {
        url += `?project=${encodeURIComponent(projectName)}`;
      }
      
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status} - ${res.statusText}`);

      const data = await res.json();

      if (data.success && Array.isArray(data.reports)) {
        let filteredReports = data.reports.slice().reverse();
        
        // Filter by project name if specified
        if (projectName) {
          filteredReports = filteredReports.filter(report => 
            report.projectName === projectName
          );
        }
        
        setReports(filteredReports);
      } else {
        setReports([]);
      }
    } catch (err) {
      console.log("Fetch error:", err);
      setError("Failed to load reports. Check connection or server.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchReports();
  }, [projectName]);

  if (loading) {
    return (
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="light-content" backgroundColor="#1e40af" />
        <ActivityIndicator size="large" color="#1e40af" />
        <Text style={styles.loadingText}>Loading reports...</Text>
      </SafeAreaView>
    );
  }

  if (error) {
    return (
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="light-content" backgroundColor="#1e40af" />
        <Text style={styles.errorText}>{error}</Text>
        <TouchableOpacity style={styles.button} onPress={fetchReports}>
          <Text style={styles.buttonText}>Retry</Text>
        </TouchableOpacity>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={[styles.container, { alignItems: "stretch" }]}>
      <StatusBar barStyle="light-content" backgroundColor="#1e40af" />
      <Text style={styles.title}>
        {projectName ? `${projectName} Reports` : 'Saved Reports'}
      </Text>
      
      <FlatList
        data={reports}
        keyExtractor={(item, index) => String(item.id || index)}
        contentContainerStyle={{ gap: 16, paddingBottom: 20, width: "100%" }}
        refreshing={loading}
        onRefresh={fetchReports}
        ListEmptyComponent={
          <Text style={styles.emptyText}>
            {projectName ? `No reports found for ${projectName}.` : 'No reports yet.'}
          </Text>
        }
        renderItem={({ item }) => {
          let displayDate = "N/A";
          if (item.submittedAt) {
            try {
              displayDate = new Date(item.submittedAt).toLocaleString();
            } catch {}
          }

          const firstPhoto = Array.isArray(item.photos) && item.photos.length > 0
            ? item.photos[0]
            : null;

          return (
            <View style={styles.reportCard}>
              {firstPhoto ? (
                <TouchableOpacity
                  onPress={() => {
                    setSelectedPhoto({
                      uri: firstPhoto,
                      datetime: item.submittedAt || 'N/A',
                      gps_latitude: item.gps || null,
                      gps_longitude: null,
                      file_size: null,
                    });
                    setModalVisible(true);
                  }}
                >
                  <Image source={{ uri: firstPhoto }} style={styles.thumb} />
                </TouchableOpacity>
              ) : (
                <View style={styles.thumbPlaceholder}>
                  <Text style={styles.thumbIcon}>📄</Text>
                </View>
              )}

              <View style={styles.reportInfo}>
                <Text style={styles.reportTitle}>
                  {item.projectName || "Untitled"}
                </Text>
                <Text style={styles.reportDate}>{String(displayDate)}</Text>
                {item.gps ? <Text style={styles.reportGPS}>📍 {String(item.gps)}</Text> : null}
                <Text style={styles.reportStatus}>
                  Status: {item.status || "Pending"}
                </Text>
                <Text style={styles.reportPhotoCount}>
                  Photos: {Array.isArray(item.photos) ? item.photos.length : 0}
                </Text>
              </View>
            </View>
          );
        }}
      />

      <Modal
        animationType="slide"
        transparent={true}
        visible={modalVisible}
        onRequestClose={() => setModalVisible(false)}
      >
        <View style={styles.modalView}>
          {selectedPhoto && (
            <>
              <Image source={{ uri: selectedPhoto.uri }} style={styles.modalImage} />
              <Text style={styles.modalText}>Datetime: {String(selectedPhoto.datetime)}</Text>
              <Text style={styles.modalText}>GPS: {String(selectedPhoto.gps_latitude || 'N/A')}</Text>
              <TouchableOpacity
                style={styles.button}
                onPress={() => setModalVisible(false)}
              >
                <Text style={styles.buttonText}>Close</Text>
              </TouchableOpacity>
            </>
          )}
        </View>
      </Modal>
    </SafeAreaView>
  );
}
// ===================== App Root =====================
// ===================== App Root =====================
const Stack = createStackNavigator();

export default function App() {
  return (
    <SafeAreaProvider>
      <LanguageProvider>
        <NavigationContainer>
          <Stack.Navigator
            initialRouteName="Login"
            screenOptions={{ headerShown: false }}
          >
            <Stack.Screen name="Login" component={LoginScreen} />
            <Stack.Screen name="ProjectSelection" component={ProjectSelectionScreen} />
            <Stack.Screen name="Main" component={MainScreen} />
            <Stack.Screen name="Report" component={ReportScreen} />
            <Stack.Screen name="SavedReports" component={SavedReportsScreen} />
          </Stack.Navigator>
        </NavigationContainer>
      </LanguageProvider>
    </SafeAreaProvider>
  );
}


// ===================== Professional Styles =====================
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#F8FAFC",
    justifyContent: "center",
    alignItems: "center",
    paddingTop: Platform.OS === "android" ? 25 : 0,
    padding: 20,
  },
  card: {
    backgroundColor: "#FFFFFF",
    padding: 24,
    borderRadius: 16,
    width: "100%",
    shadowColor: "#000",
    shadowOpacity: 0.08,
    shadowRadius: 16,
    shadowOffset: { width: 0, height: 4 },
    elevation: 8,
    marginVertical: 12,
  },
  reportCard: {
    backgroundColor: "#FFFFFF",
    padding: 20,
    borderRadius: 16,
    width: "100%",
    shadowColor: "#000",
    shadowOpacity: 0.06,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: 2 },
    elevation: 6,
    flexDirection: "row",
    gap: 16,
    alignItems: "center",
    marginVertical: 8,
    borderLeftWidth: 4,
    borderLeftColor: "#1e40af",
  },
  leaderboardCard: {
    backgroundColor: "#FFFFFF",
    padding: 20,
    borderRadius: 16,
    width: "100%",
    shadowColor: "#000",
    shadowOpacity: 0.06,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: 2 },
    elevation: 6,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginVertical: 8,
    borderLeftWidth: 4,
    borderLeftColor: "#059669",
  },
  title: {
    fontSize: 32,
    fontWeight: "700",
    marginBottom: 8,
    color: "#1e40af",
    textAlign: "center",
  },
  subtitle: {
    fontSize: 16,
    color: "#64748B",
    marginBottom: 20,
    textAlign: "center",
    fontWeight: "400",
  },
  headerWrap: {
    width: "100%",
    alignItems: "center",
    marginBottom: 32,
    position: 'relative',
  },
  projectDisplay: {
    backgroundColor: "#EEF2FF",
    padding: 12,
    borderRadius: 12,
    marginVertical: 8,
    borderLeftWidth: 4,
    borderLeftColor: "#1e40af",
    width: "100%",
  },
  projectLabel: {
    fontSize: 14,
    fontWeight: "600",
    color: "#64748B",
    marginBottom: 4,
  },
  projectValue: {
    fontSize: 16,
    fontWeight: "700",
    color: "#1e40af",
  },
  backButton: {
    position: "absolute",
    top: -20,
    left: 0,
    backgroundColor: "#10b981",
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 8,
    shadowColor: "#000",
    shadowOpacity: 0.2,
    shadowRadius: 4,
    shadowOffset: { width: 0, height: 2 },
    elevation: 4,
  },
  backButtonText: {
    color: "#FFFFFF",
    fontWeight: "600",
    fontSize: 14,
  },
  label: {
    fontSize: 18,
    marginTop: 16,
    marginBottom: 8,
    color: "#0F172A",
    fontWeight: "600",
  },
  input: {
    borderWidth: 1.5,
    borderColor: "#D1D5DB",
    padding: 16,
    marginBottom: 16,
    borderRadius: 12,
    backgroundColor: "#FFFFFF",
    width: "100%",
    fontSize: 16,
    color: "#0F172A",
    shadowColor: "#000",
    shadowOpacity: 0.04,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 2 },
    elevation: 2,
  },
  button: {
    backgroundColor: "#1e40af",
    paddingVertical: 16,
    paddingHorizontal: 24,
    borderRadius: 12,
    marginVertical: 12,
    alignItems: "center",
    shadowColor: "#1e40af",
    shadowOpacity: 0.3,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: 4 },
    elevation: 8,
    width: "100%",
  },
  buttonText: {
    color: "#FFFFFF",
    fontWeight: "700",
    fontSize: 16,
    letterSpacing: 0.5,
  },
  image: {
    width: "100%",
    height: 240,
    marginTop: 16,
    borderRadius: 12,
  },
  preview: {
    width: 120,
    height: 120,
    borderRadius: 12,
    backgroundColor: "#F1F5F9",
    borderWidth: 2,
    borderColor: "#E2E8F0",
  },
  thumb: {
    width: 64,
    height: 64,
    borderRadius: 12,
    backgroundColor: "#F1F5F9",
    borderWidth: 1.5,
    borderColor: "#E2E8F0",
  },
  thumbPlaceholder: {
    width: 64,
    height: 64,
    borderRadius: 12,
    backgroundColor: "#F1F5F9",
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 1.5,
    borderColor: "#E2E8F0",
  },
  thumbIcon: {
    fontSize: 24,
  },
  bigLogo: {
    width: 120,
    height: 120,
    marginBottom: 16,
    borderRadius: 24,
    shadowColor: "#000",
    shadowOpacity: 0.1,
    shadowRadius: 16,
    shadowOffset: { width: 0, height: 4 },
    elevation: 8,
  },
  smallLogo: {
    width: 48,
    height: 48,
    marginBottom: 12,
    borderRadius: 12,
    shadowColor: "#000",
    shadowOpacity: 0.1,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 2 },
    elevation: 4,
  },
  gps: {
    marginTop: 12,
    fontSize: 16,
    fontStyle: "italic",
    color: "#059669",
    fontWeight: "500",
    textAlign: "center",
    backgroundColor: "#ECFDF5",
    padding: 12,
    borderRadius: 8,
  },
  modalView: {
    flex: 1,
    backgroundColor: "#FFFFFF",
    marginTop: 60,
    margin: 20,
    borderRadius: 16,
    padding: 24,
    alignItems: "center",
    shadowColor: "#000",
    shadowOpacity: 0.15,
    shadowRadius: 20,
    shadowOffset: { width: 0, height: 8 },
    elevation: 12,
  },
  modalImage: {
    width: "100%",
    height: 300,
    borderRadius: 12,
    marginBottom: 20,
  },
  modalText: {
    fontSize: 16,
    color: "#374151",
    marginBottom: 8,
    fontWeight: "500",
  },
  loginCard: {
    backgroundColor: "#FFFFFF",
    padding: 32,
    borderRadius: 20,
    width: "100%",
    maxWidth: 400,
    alignItems: "center",
    shadowColor: "#000",
    shadowOpacity: 0.1,
    shadowRadius: 20,
    shadowOffset: { width: 0, height: 8 },
    elevation: 12,
  },
  toggleContainer: {
    flexDirection: "row",
    marginBottom: 24,
    backgroundColor: "#F1F5F9",
    borderRadius: 12,
    padding: 4,
    width: "100%",
  },
  toggleButton: {
    flex: 1,
    paddingVertical: 12,
    alignItems: "center",
    borderRadius: 8,
  },
  toggleActive: {
    backgroundColor: "#1e40af",
    shadowColor: "#1e40af",
    shadowOpacity: 0.3,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 2 },
    elevation: 4,
  },
  toggleText: {
    color: "#64748B",
    fontWeight: "600",
    fontSize: 16,
  },
  toggleActiveText: {
    color: "#FFFFFF",
  },
  reportInfo: {
    flex: 1,
  },
  reportTitle: {
    fontWeight: "700",
    fontSize: 18,
    color: "#0F172A",
    marginBottom: 4,
  },
  reportDate: {
    color: "#64748B",
    fontSize: 14,
    marginBottom: 4,
    fontWeight: "400",
  },
  reportGPS: {
    color: "#059669",
    fontSize: 14,
    marginBottom: 4,
    fontWeight: "500",
  },
  reportStatus: {
    color: "#059669",
    fontSize: 14,
    marginBottom: 4,
    fontWeight: "600",
  },
  reportPhotoCount: {
    color: "#64748B",
    fontSize: 14,
    fontWeight: "500",
  },
  leaderboardRank: {
    fontWeight: "700",
    fontSize: 18,
    color: "#0F172A",
  },
  leaderboardPoints: {
    fontSize: 18,
    color: "#1e40af",
    fontWeight: "700",
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
    color: "#64748B",
    fontWeight: "500",
  },
  errorText: {
    color: "#EF4444",
    marginTop: 20,
    fontSize: 16,
    textAlign: "center",
    fontWeight: "500",
  },
  emptyText: {
    textAlign: "center",
    marginTop: 40,
    fontSize: 16,
    color: "#64748B",
    fontWeight: "500",
  },
  logoutButton: {
    position: "absolute",
    top: -20,
    right: 0,
    backgroundColor: "#1e40af", 
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 8,
    shadowColor: "#000",
    shadowOpacity: 0.2,
    shadowRadius: 4,
    shadowOffset: { width: 0, height: 2 },
    elevation: 4,
  },
  logoutText: {
    color: "#FFFFFF",
    fontWeight: "600",
    fontSize: 14,
  },
});


// App Registry
AppRegistry.registerComponent('main',()=>App)