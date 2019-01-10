#include <iostream>
#include <stack>

#include <Eigen/QR>

#include <yarp/os/LogStream.h>
#include <yarp/os/ResourceFinder.h>
#include <yarp/sig/Vector.h>

#include <iDynTree/InverseKinematics.h>
#include <iDynTree/KinDynComputations.h>
#include <iDynTree/Core/Transform.h>
#include <iDynTree/Core/TestUtils.h>
#include <iDynTree/Model/JointState.h>
#include <iDynTree/ModelIO/ModelLoader.h>
#include <iDynTree/Model/Traversal.h>

/*!
 * Relevant information on the submodel between two links (segments)
 * Needed to compute inverse kinematics, velocities, etc
 */
struct LinkPairInfo {
    // Variables representing the DoFs between the two frames
    iDynTree::VectorDynSize jointConfigurations;
    iDynTree::VectorDynSize jointVelocities;

    // Transformation variable
    iDynTree::Transform relativeTransformation; // TODO: If this is wrt global frame

    // IK elements (i.e. compute joints)
    iDynTree::InverseKinematics ikSolver;

    // Initial joint positions
    iDynTree::VectorDynSize sInitial;

    // Reduced model of link pair
    iDynTree::Model pairModel;

    // Velocity-related elements
    iDynTree::MatrixDynSize parentJacobian;
    iDynTree::MatrixDynSize childJacobian;
    iDynTree::MatrixDynSize relativeJacobian;
    std::unique_ptr<iDynTree::KinDynComputations> kinDynComputations;
    Eigen::ColPivHouseholderQR<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > jacobianDecomposition;

    // Mapping from link pair to full model. Needed to map from small to complete problem
    std::string parentFrameName; //name of the parent frame
    iDynTree::FrameIndex parentFrameModelIndex; //index of the frame in the iDynTree Model
    iDynTree::FrameIndex parentFrameSegmentsIndex; //index of the parent frame in the segment list

    std::string childFrameName; //name of the child frame
    iDynTree::FrameIndex childFrameModelIndex; //index of the frame in the iDynTree Model
    iDynTree::FrameIndex childFrameSegmentsIndex; //index of the child frame in the segment list

    std::vector<std::pair<size_t, size_t> > consideredJointLocations; /*!< For each joint connecting the pair: first = offset in the full model , second = dofs of joint */

    LinkPairInfo() = default;
#if defined(_MSC_VER) && _MSC_VER < 1900
    LinkPairInfo(LinkPairInfo&& rvalue)
    : jointConfigurations(std::move(rvalue.jointConfigurations))
    , jointVelocities(std::move(rvalue.jointVelocities))
    , ikSolver(std::move(rvalue.ikSolver))
    , parentJacobian(std::move(rvalue.parentJacobian))
    , childJacobian(std::move(rvalue.childJacobian))
    , relativeJacobian(std::move(rvalue.relativeJacobian))
    , kinDynComputations(std::move(rvalue.kinDynComputations))
    , jacobianDecomposition(std::move(rvalue.jacobianDecomposition))
    , parentFrameName(std::move(rvalue.parentFrameName))
    , parentFrameModelIndex(rvalue.parentFrameModelIndex)
    , parentFrameSegmentsIndex(rvalue.parentFrameSegmentsIndex)
    , childFrameName(std::move(rvalue.childFrameName))
    , childFrameModelIndex(rvalue.childFrameModelIndex)
    , childFrameSegmentsIndex(rvalue.childFrameSegmentsIndex)
    , consideredJointLocations(std::move(rvalue.consideredJointLocations))
    {}
#else
    LinkPairInfo(LinkPairInfo&&) = default;
#endif
    LinkPairInfo(const LinkPairInfo&) = delete;

};

// Relevant information on the segment input
struct SegmentInfo {
    std::string segmentName;

    iDynTree::Transform poseWRTWorld;
    iDynTree::VectorDynSize velocities;

    // TODO: if not needed acceleration delete them
    yarp::sig::Vector accelerations;
};

/*!
 * @brief analyze model and list of segments to create all possible segment pairs
 *
 * @param[in] model the full model
 * @param[in] humanSegments list of segments on which look for the possible pairs
 * @param[out] framePairs resulting list of all possible pairs. First element is parent, second is child
 * @param[out] framePairIndeces indeces in the humanSegments list of the pairs in framePairs
 */
static void createEndEffectorsPairs(const iDynTree::Model& model,
                                    std::vector<SegmentInfo>& humanSegments,
                                    std::vector<std::pair<std::string, std::string>> &framePairs,
                                    std::vector<std::pair<iDynTree::FrameIndex, iDynTree::FrameIndex> > &framePairIndeces);

static bool getReducedModel(const iDynTree::Model& modelInput,
                            const std::string& parentFrame,
                            const std::string& endEffectorFrame,
                            iDynTree::Model& modelOutput);

using namespace std;

int main()
{
    std::cout << "Ipopt testing..." << std::endl;

    // ==========================
    // INITIALIZE THE HUMAN MODEL
    // ==========================

    std::string urdfFileName = "Claudia66DoF.urdf";

    auto& rf = yarp::os::ResourceFinder::getResourceFinderSingleton();
    std::string urdfFilePath = rf.findFile(urdfFileName);
    if (urdfFilePath.empty()) {
        yError() << "Failed to find file" << urdfFileName;
        return -1;
    }

    iDynTree::ModelLoader modelLoader;
    if (!modelLoader.loadModelFromFile(urdfFilePath) || !modelLoader.isValid()) {
        yError() << "Failed to load model" << urdfFilePath;
        return -1;
    }

    iDynTree::Model humanModel = modelLoader.model();
    //yInfo() << "Given human model:";
    //yInfo() << humanModel.toString();

    // ================================
    // INITIALIZE LINK / JOINTS BUFFERS
    // ================================

    // Setting the segment names explicitly without passing through configuration
    const int nrOfSegments = 23;
    std::vector<std::string> segmentsNameList;
    segmentsNameList.resize(nrOfSegments);

    segmentsNameList.at(0) = "Pelvis";
    segmentsNameList.at(1) = "L5";
    segmentsNameList.at(2) = "L3";
    segmentsNameList.at(3) = "T12";
    segmentsNameList.at(4) = "T8";
    segmentsNameList.at(5) = "Neck";
    segmentsNameList.at(6) = "Head";
    segmentsNameList.at(7) = "RightShoulder";
    segmentsNameList.at(8) = "RightUpperArm";
    segmentsNameList.at(9) = "RightForeArm";
    segmentsNameList.at(10) = "RightHand";
    segmentsNameList.at(11) = "LeftShoulder";
    segmentsNameList.at(12) = "LeftUpperArm";
    segmentsNameList.at(13) = "LeftForeArm";
    segmentsNameList.at(14) = "LeftHand";
    segmentsNameList.at(15) = "RightUpperLeg";
    segmentsNameList.at(16) = "RightLowerLeg";
    segmentsNameList.at(17) = "RightFoot";
    segmentsNameList.at(18) = "RightToe";
    segmentsNameList.at(19) = "LeftUpperLeg";
    segmentsNameList.at(20) = "LeftLowerLeg";
    segmentsNameList.at(21) = "LeftFoot";
    segmentsNameList.at(22) = "LeftToe";

    // Create a vector of segment struct instances
    std::vector<SegmentInfo> segments;
    segments.resize(nrOfSegments);

    // Fill the segments info based on segmentsNameList entries
    for (size_t index = 0; index < segmentsNameList.size(); index++) {
        segments[index].segmentName = segmentsNameList.at(index);
        segments[index].velocities.resize(6);
        segments[index].velocities.zero();
    }

    // Variables for link pairs
    std::vector<std::pair<std::string, std::string>> pairNames;
    std::vector<std::pair<iDynTree::FrameIndex, iDynTree::FrameIndex> > pairSegmentIndeces;

    // Get the pairNames
    createEndEffectorsPairs(humanModel, segments, pairNames, pairSegmentIndeces);

    // Create an vector of LinkPairInfo instances
    std::vector<LinkPairInfo> linkPairs;
    linkPairs.resize(pairNames.size()); // This will be of size segments-1 i.e 22
    yInfo() << "link pairs size : " << linkPairs.size();

    // Initialize the linkPari instances
    for (unsigned index = 0; index < pairNames.size(); ++index) {

        LinkPairInfo& pairInfo = linkPairs[index];

        pairInfo.parentFrameName = pairNames[index].first;
        pairInfo.parentFrameSegmentsIndex = pairSegmentIndeces[index].first;

        pairInfo.childFrameName = pairNames[index].second;
        pairInfo.childFrameSegmentsIndex = pairSegmentIndeces[index].second;

        // Get the reduced model for the pait
        if (!getReducedModel(humanModel, pairInfo.parentFrameName, pairInfo.childFrameName, pairInfo.pairModel)) {

            yWarning() << "failed to get reduced model for the segment pair " << pairInfo.parentFrameName.c_str()
                       << ", " << pairInfo.childFrameName.c_str();
            continue;
        }

        // Resize initial joint positions size
        pairInfo.sInitial.resize(pairInfo.pairModel.getNrOfJoints());

        // Joints that will be solved by Ipopt
        std::vector<std::string> solverJoints;
        solverJoints.resize(pairInfo.pairModel.getNrOfJoints());

        for (size_t i=0; i < pairInfo.pairModel.getNrOfJoints(); i++) {
            solverJoints[i] = pairInfo.pairModel.getJointName(i);
        }

        pairInfo.consideredJointLocations.reserve(solverJoints.size());
        for (auto &jointName: solverJoints) {
            iDynTree::JointIndex jointIndex = humanModel.getJointIndex(jointName);
            if (jointIndex == iDynTree::JOINT_INVALID_INDEX) {
                yWarning() << "IK considered joint " << jointName << " not found in the complete model";
                continue;
            }
            iDynTree::IJointConstPtr joint = humanModel.getJoint(jointIndex);

            // Save location and length of each DoFs
            pairInfo.consideredJointLocations.push_back(std::pair<size_t, size_t>(joint->getDOFsOffset(), joint->getNrOfDOFs()));
        }

        pairInfo.jointConfigurations.resize(solverJoints.size());
        pairInfo.jointConfigurations.zero();

        // Same size and initialization
        pairInfo.jointVelocities = pairInfo.jointConfigurations;

    }

    {
        yInfo() << "***********************************";
        yInfo() << "Solving for a single link pair...";
        yInfo() << "***********************************";

        // Single linkPair processing
        LinkPairInfo& linkPair = linkPairs.at(0);

        // Set known tranforms for parent and child links
        iDynTree::Position parentPos(1,1,1);
        iDynTree::Rotation parentRot(1,0,0,0,1,0,0,0,1);
        iDynTree::Transform parentTransform(parentRot, parentPos);

        iDynTree::Position childPos(1,1,1.5);
        iDynTree::Rotation childRot(-1,0,0,0,-1,0,0,0,1); // Set this to be rotation about z by pi
        iDynTree::Transform childTransform(childRot, childPos);

        // Set initial joint positions to zero
        linkPair.sInitial.zero();
        yInfo() << "Initial joint positions : " << linkPair.sInitial.toString();

        // Create an instance of ik
        iDynTree::InverseKinematics ik;

        // Set ik parameters
        ik.setVerbosity(1);
        ik.setLinearSolverName("ma27");
        ik.setDefaultTargetResolutionMode(iDynTree::InverseKinematicsTreatTargetAsConstraintNone);
        ik.setRotationParametrization(iDynTree::InverseKinematicsRotationParametrizationRollPitchYaw);

        // Set ik model
        if (!ik.setModel(linkPair.pairModel)) {
            yWarning() << "failed to configure IK solver for the segment pair" << linkPair.parentFrameName.c_str()
                       << ", " << linkPair.childFrameName.c_str() <<  " Skipping pair";
        }

        // Add parent link as fixed base constraint with identity transform
        ik.addFrameConstraint(linkPair.parentFrameName, iDynTree::Transform::Identity());

        // Add child link as a target and set initial transform to be identity
        ik.addTarget(linkPair.childFrameName, iDynTree::Transform::Identity());

        // Set initial conditions
        ik.setFullJointsInitialCondition(&(parentTransform), &(linkPair.sInitial));

        // Get the relative transform
        iDynTree::Transform parent_H_target = parentTransform.inverse() * childTransform;

        // Update the child target
        ik.updateTarget(linkPair.childFrameName, parent_H_target);

        int result = ik.solve();

        // Get optimization solution
        iDynTree::Transform outputTransform; // This name is not clear
        iDynTree::VectorDynSize jointConfigurationSolution;
        jointConfigurationSolution.resize(linkPair.sInitial.size());

        ik.getFullJointsSolution(outputTransform, jointConfigurationSolution);
        yInfo() << "parent name : " << linkPair.parentFrameName << " , child name : " << linkPair.childFrameName;
        yInfo() << "IK Result : " << result << " ,Joint configuration : " << linkPair.jointConfigurations.toString();

    }

    {
        yInfo() << "************************************";
        yInfo() << "Solving for a all the link pairs...";
        yInfo() << "************************************";

        for (auto& linkPair : linkPairs) {

            // Set known tranforms for parent and child links
            iDynTree::Position parentPos(1,1,1);
            iDynTree::Rotation parentRot(1,0,0,0,1,0,0,0,1);
            iDynTree::Transform parentTransform(parentRot, parentPos);

            iDynTree::Position childPos(1,1,1.5);
            iDynTree::Rotation childRot(-1,0,0,0,-1,0,0,0,1); // Set this to be rotation about z by pi
            iDynTree::Transform childTransform(childRot, childPos);

            // Set initial joint positions to zero
            linkPair.sInitial.zero();
            yInfo() << "Initial joint positions : " << linkPair.sInitial.toString();

            // Create an instance of ik
            iDynTree::InverseKinematics ik;

            // Set ik parameters
            ik.setVerbosity(1);
            ik.setLinearSolverName("ma27");
            ik.setDefaultTargetResolutionMode(iDynTree::InverseKinematicsTreatTargetAsConstraintNone);
            ik.setRotationParametrization(iDynTree::InverseKinematicsRotationParametrizationRollPitchYaw);

            // Set ik model
            if (!ik.setModel(linkPair.pairModel)) {
                yWarning() << "failed to configure IK solver for the segment pair" << linkPair.parentFrameName.c_str()
                           << ", " << linkPair.childFrameName.c_str() <<  " Skipping pair";
            }

            // Add parent link as fixed base constraint with identity transform
            ik.addFrameConstraint(linkPair.parentFrameName, iDynTree::Transform::Identity());

            // Add child link as a target and set initial transform to be identity
            ik.addTarget(linkPair.childFrameName, iDynTree::Transform::Identity());

            // Set initial conditions
            ik.setFullJointsInitialCondition(&(parentTransform), &(linkPair.sInitial));

            // Get the relative transform
            iDynTree::Transform parent_H_target = parentTransform.inverse() * childTransform;

            // Update the child target
            ik.updateTarget(linkPair.childFrameName, parent_H_target);

            int result = ik.solve();

            // Get optimization solution
            iDynTree::Transform outputTransform; // This name is not clear
            iDynTree::VectorDynSize jointConfigurationSolution;
            jointConfigurationSolution.resize(linkPair.sInitial.size());

            ik.getFullJointsSolution(outputTransform, jointConfigurationSolution);
            yInfo() << "parent name : " << linkPair.parentFrameName << " , child name : " << linkPair.childFrameName;
            yInfo() << "IK Result : " << result << " ,Joint configuration : " << linkPair.jointConfigurations.toString();

        }
    }

    return 0;


}

// This method returns the link pari names from the human model
static void createEndEffectorsPairs(const iDynTree::Model& model,
                                    std::vector<SegmentInfo>& humanSegments,
                                    std::vector<std::pair<std::string, std::string> > &framePairs,
                                    std::vector<std::pair<iDynTree::FrameIndex, iDynTree::FrameIndex> > &framePairIndeces)
{
    //for each element in human segments
    //extract it from the vector (to avoid duplications)
    //Look for it in the model and get neighbours
    std::vector<SegmentInfo> segments(humanSegments);
    size_t segmentCount = segments.size();

    while (!segments.empty()) {
        SegmentInfo segment = segments.back();
        segments.pop_back();
        segmentCount--;

        iDynTree::LinkIndex linkIndex = model.getLinkIndex(segment.segmentName);
        if (linkIndex < 0 || static_cast<unsigned>(linkIndex) >= model.getNrOfLinks()) {
            yWarning("Segment %s not found in the URDF model", segment.segmentName.c_str());
            continue;
        }

        //this for loop should not be necessary, but this can help keeps the backtrace short
        //as we do not assume that we can go back further that this node
        for (unsigned neighbourIndex = 0; neighbourIndex < model.getNrOfNeighbors(linkIndex); ++neighbourIndex) {
            //remember the "biforcations"
            std::stack<iDynTree::LinkIndex> backtrace;
            //and the visited nodes
            std::vector<iDynTree::LinkIndex> visited;

            //I've already visited the starting node
            visited.push_back(linkIndex);
            iDynTree::Neighbor neighbour = model.getNeighbor(linkIndex, neighbourIndex);
            backtrace.push(neighbour.neighborLink);

            while (!backtrace.empty()) {
                iDynTree::LinkIndex currentLink = backtrace.top();
                backtrace.pop();
                //add the current link to the visited
                visited.push_back(currentLink);

                std::string linkName = model.getLinkName(currentLink);

                // check if this is a human segment
                std::vector<SegmentInfo>::iterator foundSegment = std::find_if(segments.begin(),
                                                                               segments.end(),
                                                                               [&](SegmentInfo& frame){ return frame.segmentName == linkName; });
                if (foundSegment != segments.end()) {
                    std::vector<SegmentInfo>::difference_type foundLinkIndex = std::distance(segments.begin(), foundSegment);
                    //Found! This is a segment
                    framePairs.push_back(std::pair<std::string, std::string>(segment.segmentName, linkName));
                    framePairIndeces.push_back(std::pair<iDynTree::FrameIndex, iDynTree::FrameIndex>(segmentCount, foundLinkIndex));
                    break;
                }
                //insert all non-visited neighbours
                for (unsigned i = 0; i < model.getNrOfNeighbors(currentLink); ++i) {
                    iDynTree::LinkIndex link = model.getNeighbor(currentLink, i).neighborLink;
                    //check if we already visited this segment
                    if (std::find(visited.begin(), visited.end(), link) != visited.end()) {
                        //Yes => skip
                        continue;
                    }
                    backtrace.push(link);
                }
            }

        }
    }

}

static bool getReducedModel(const iDynTree::Model& modelInput,
                            const std::string& parentFrame,
                            const std::string& endEffectorFrame,
                            iDynTree::Model& modelOutput)
{
    iDynTree::FrameIndex parentFrameIndex;
    iDynTree::FrameIndex endEffectorFrameIndex;
    std::vector<std::string> consideredJoints;
    iDynTree::Traversal traversal;
    iDynTree::LinkIndex parentLinkIdx;
    iDynTree::IJointConstPtr joint;
    iDynTree::ModelLoader loader;

    // Get frame indices
    parentFrameIndex = modelInput.getFrameIndex(parentFrame);
    endEffectorFrameIndex = modelInput.getFrameIndex(endEffectorFrame);

    if(parentFrameIndex == iDynTree::FRAME_INVALID_INDEX){
        yError() << " Invalid parent frame: "<< parentFrame;
        return false;
    }
    else if(endEffectorFrameIndex == iDynTree::FRAME_INVALID_INDEX){
        yError() << " Invalid End Effector Frame: "<< endEffectorFrame;
        return false;
    }

    // Select joint from traversal
    modelInput.computeFullTreeTraversal(traversal, modelInput.getFrameLink(parentFrameIndex));

    iDynTree::LinkIndex visitedLink = modelInput.getFrameLink(endEffectorFrameIndex);

    while (visitedLink != traversal.getBaseLink()->getIndex())
    {        
        parentLinkIdx = traversal.getParentLinkFromLinkIndex(visitedLink)->getIndex();
        joint = traversal.getParentJointFromLinkIndex(visitedLink);

        // Check if the joint is supported
        if(modelInput.getJoint(joint->getIndex())->getNrOfDOFs() == 1)
        {
            consideredJoints.insert(consideredJoints.begin(), modelInput.getJointName(joint->getIndex()));
        }
        else {
            yWarning() << "Joint " << modelInput.getJointName(joint->getIndex()) << " is ignored as it has (" << modelInput.getJoint(joint->getIndex())->getNrOfDOFs() << " DOFs)";
        }

        visitedLink = parentLinkIdx;
    }

    if (!loader.loadReducedModelFromFullModel(modelInput, consideredJoints)) {
        std::cerr << " failed to select joints: " ;
        for (std::vector< std::string >::const_iterator i = consideredJoints.begin(); i != consideredJoints.end(); ++i){
            std::cerr << *i << ' ';
        }
        std::cerr << std::endl;
        return false;

    }

    modelOutput = loader.model();

    return true;
}

