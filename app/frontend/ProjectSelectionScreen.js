import React, { useState } from 'react';
import { 
  View, 
  Text, 
  TouchableOpacity, 
  StyleSheet, 
  StatusBar,
  FlatList,
  Image
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useLanguage } from './LanguageContext';
import AppLogo from './assets/images/app-logo.jpg';

const ProjectSelectionScreen = ({ route, navigation }) => {
  const { name, agentId } = route.params || {};
  const { t } = useLanguage();
  const [selectedProject, setSelectedProject] = useState(null);

  // Sample projects - you can modify this or fetch from your server
   const projects = [
    {
      id: 1,
      name: "Mangrove Restoration Phase1",
      description: "Water Quality Assessment - Phase 1",
      status: "Active",
      color: "#1e40af"
    },
    {
      id: 2,
      name: "Coral Reef Protection Initiative",
      description: "Environmental Monitoring - Urban Areas",
      status: "Pending",
      color: "#059669"
    },
    {
      id: 3,
      name: "Seagrass Meadow Restoration ",
      description: "Coastal Water Analysis - Research",
      status: "Completed",
      color: "#7c3aed"
    },
    {
      id: 4,
      name: "Mangrove Restoration Phase2",
      description: "Industrial Discharge Monitoring",
      status: "Active",
      color: "#dc2626"
    }
  ];

  const handleProjectSelect = (project) => {
    setSelectedProject(project);
    // Navigate to Main screen with selected project
    navigation.replace("Main", { 
      name, 
      agentId, 
      project: project.name,
      projectName: project.name,
      projectId: project.id 
    });
  };

  const handleLogout = () => {
    navigation.replace("Login");
  };

  const renderProjectCard = ({ item }) => (
    <TouchableOpacity
      style={[styles.projectCard, { borderLeftColor: item.color }]}
      onPress={() => handleProjectSelect(item)}
      activeOpacity={0.7}
    >
      <View style={styles.projectContent}>
        <View style={styles.projectHeader}>
          <Text style={styles.projectName}>{item.name}</Text>
          <View style={[styles.statusBadge, { 
            backgroundColor: item.status === 'Active' ?'#3b82f6' : 
                           item.status === 'Completed' ? '#10b981' : '#f59e0b' 
          }]}>
            <Text style={styles.statusText}>{item.status}</Text>
          </View>
        </View>
        <Text style={styles.projectDescription}>{item.description}</Text>
        <View style={styles.projectFooter}>
          <Text style={styles.selectText}>Tap to select →</Text>
        </View>
      </View>
    </TouchableOpacity>
  );

  return (
    <SafeAreaView style={styles.container} edges={['right', 'bottom', 'left']}>
      <StatusBar barStyle="light-content" backgroundColor="#1e40af" />
      
      {/* Logout button positioned at top */}
      <TouchableOpacity style={styles.logoutButton} onPress={handleLogout}>
        <Text style={styles.logoutText}>Logout</Text>
      </TouchableOpacity>

      {/* Centered Content Container */}
      <View style={styles.centeredContainer}>
        {/* Header */}
        <View style={styles.headerWrap}>
          <Image source={AppLogo} style={styles.smallLogo} />
          <Text style={styles.title}>
            Welcome, {name || "User"} 👋
          </Text>
          <Text style={styles.subtitle}>
            Agent ID: {agentId || "N/A"}
          </Text>
          <Text style={styles.subtitle}>
            Select a project to continue
          </Text>
        </View>

        {/* Project List */}
        <View style={styles.projectListContainer}>
          <Text style={styles.sectionTitle}>Available Projects</Text>
          <FlatList
            data={projects}
            renderItem={renderProjectCard}
            keyExtractor={(item) => item.id.toString()}
            contentContainerStyle={styles.projectList}
            showsVerticalScrollIndicator={false}
          />
        </View>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#F8FAFC",
    padding: 20,
  },
  centeredContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 20,
  },
  headerWrap: {
    width: "100%",
    alignItems: "center",
    marginBottom: 24,
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
  title: {
    fontSize: 26,
    fontWeight: "700",
    marginBottom: 8,
    color: "#1e40af",
    textAlign: "center",
  },
  subtitle: {
    fontSize: 15,
    color: "#64748B",
    marginBottom: 6,
    textAlign: "center",
    fontWeight: "400",
  },
  logoutButton: {
    position: "absolute",
    top: 10,
    right: 10,
    backgroundColor: "#ef4444",
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 8,
    shadowColor: "#000",
    shadowOpacity: 0.2,
    shadowRadius: 4,
    shadowOffset: { width: 0, height: 2 },
    elevation: 4,
    zIndex: 10,
  },
  logoutText: {
    color: "#FFFFFF",
    fontWeight: "600",
    fontSize: 13,
  },
  projectListContainer: {
    flex: 1,
    width: "100%",
    maxHeight: 400, // Limit height to prevent overflow
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: "700",
    color: "#0F172A",
    marginBottom: 12,
    textAlign: "center",
  },
  projectList: {
    paddingBottom: 20,
  },
  projectCard: {
    backgroundColor: "#FFFFFF",
    marginVertical: 6,
    borderRadius: 16,
    shadowColor: "#000",
    shadowOpacity: 0.06,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: 2 },
    elevation: 6,
    borderLeftWidth: 4,
    overflow: 'hidden',
  },
  projectContent: {
    padding: 16,
  },
  projectHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  projectName: {
    fontSize: 16,
    fontWeight: "700",
    color: "#0F172A",
    flex: 1,
    marginRight: 8,
  },
  statusBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
    minWidth: 70,
    alignItems: 'center',
  },
  statusText: {
    color: "#FFFFFF",
    fontSize: 11,
    fontWeight: "600",
  },
  projectDescription: {
    fontSize: 13,
    color: "#64748B",
    marginBottom: 10,
    lineHeight: 18,
  },
  projectFooter: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
  },
  selectText: {
    fontSize: 13,
    color: "#1e40af",
    fontWeight: "600",
  },
});

export default ProjectSelectionScreen;