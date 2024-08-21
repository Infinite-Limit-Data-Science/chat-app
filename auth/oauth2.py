from fastapi.security import OAuth2PasswordBearer

### Custom applications can use Entra to manage the identity of their users and perform authentication through Entra. 
# An Entra Tenant represents a dedicated instance of Microsoft Entra ID for your organization. It is essentially your 
# organization’s space in the Azure cloud where you manage your identities and access. Note in additional to the default
# workforce Tenant, with a subscription, you can create an Azure AD B2C tenant. (External Tenant). A workforce tenant 
# configuration is for your employees, internal business apps, and other organizational resources. a directory is a 
# component of a tenant that contains the actual data about users, groups, and other entities. While it's true that 
# a tenant can have multiple Entra B2C directories, these directories are not the same as the main Entra ID Directory.

# When you create a Tenant, you choose a Tenant Name and that Tenant Name is used as part of the fully qualified domain 
# name for the Tenant (e.g. example.onmicrosoft.com). By default, that example.onmicrosoft.com domain is known as the 
# Primary Domain. A question is where does this domain come into play? When you add a new User to your Tenant, you must 
# specify a "User name", and it will be linked with the Domain provisioned for you when you created the Tenant. So if 
# your company is example.com, by default, you cannot use user@example.com. Instead it will look something like 
# userexample@onmicrosoft.com. To fix this, go to Entra ID portal (which loads your default Tenant) > Manage > "Custom 
# domain names" > "Add Custom Domain". To prove you own the Domain, you need to copy the TXT or MX record in Azure and 
# add it to the DNS Server (such as AWS Route 53) into the doman's Hosted Zone.

# Every Azure Account/User is part of at least one Tenant (organization). The identities that Entra ID manages are both 
# Users and Applications (since Applications can have identities). In the case of Users, we are talking about individuals 
# who have been granted access to some resources within Entra. In Entra ID portal, visit Manage > Users > "Create new user"
# OR "Invite external user" to be part of your Tenant.

# The main identity of a User is the "User principal name". The "User principal name" is a name you assign to the User 
# concatenated with one of the domain names associated with the Tenant via the "Add custom domains" portal. This could 
# either be the default userexample.onmicrosoft.com (the onmicrosoft address exists for every Tenant) or a custom domain 
# you created such as example.com and so the "User name" would be, for example, user@example.com. Importantly, in the 
# Entra ID > "Custom Domain Names" portal, you can have multiple custom domain names, such as atlanticgenetics.com and 
# atlanticlabs.net along with the default atlanticgenetics.onmicrosoft.com email, for example. So when creating a new 
# user, you can set any of these domains as part of the "User principal name".

# When creating a new user, there are two additional tabs, "Properties" and "Assignments". Under Properties, specify the 
# user type, Member or Guest.

# When creating a new user, you can pick a domain name that is configured in the "Add Custom Domains" subportal when setting 
# the "User principal name" option. But you may want to assign access to your Tenant from someone outside of one of your domains, 
# such as a business partner or external user (a contractor that works from home). These are examples of individuals who do not 
# have an email address within your "Add Custom Domains" domains. The email is important. This is how business partners and 
# contractors will reset their password, and this is how you will give them access to your applications within Entra ID. The 
# concept of inviting external users is called a Guest User. So when creating a User in the Users portal, instead of selecting 
# "Create user", you specify "Invite user". The Guest User will be emailed an invitation they can accept to begin collaborating.
#  You can assign them Groups. You can assign them Roles, such as the Application Developer role. They can still go through MFA 
# or Conditional Access. Thus, they can still follow the same security standards as your own Users.

# In Entra ID, a Group allows you to grant access and permissions to a group of users instead of for each individual user. Limiting 
# access to Microsoft Entra resources to only those users who need access is one of the core security principles of Zero Trust.

# A Group requires a Group Type and Group name. The Group Type can either be Security or Microsoft 365. While both types can be 
# used for grouping users, they serve different purposes. A Security group is a traditional group type used for access control 
# of Entra resources. A Microsoft 365 group, on the other hand, is a type of group specifically designed for Microsoft 365 
# services, such as Office 365, Teams, and SharePoint. 

# You can add Users or Guests to a Group if the User Type is Member or Guest. By default, assigning Users to Group does not grant 
# them any new permissions. You can assign Roles to the Group, which allow for managing the Entra ID Tenant, including managing 
# Entra users, Entra groups, and Entra applications.

# Another type of Group is the Dynamic Group. When creating a new Group under Manage > Groups > New Group, the Default Group will 
# have an "Assigned" Membership. You change Membership Type in Console to "Dynamic User" instead of "Assigned", which gives you a 
# Dynamic Group. Note using Dynamic Groups requires a Microsoft Entra ID P1 license. Also, note the difference between "Assigned" 
# and "Dynamic Groups". An Assigned group is a traditional group where membership is manually managed by an administrator. Members 
# are added or removed explicitly by the administrator, and the group membership is static. A Dynamic group is a group where 
# membership is determined by a set of rules based on user attributes, such as department, job title, location, or other 
# properties. The group membership is dynamic and updates automatically when user attributes change.

# Microsoft Entra ID P2 - Microsoft Entra ID P2 is available as a standalone product or included with Microsoft 365 E5 for 
# enterprise customers

# Entra Licenses are assigned to Entra Users or Entra Groups. To assign Licenses, in the Entra ID portal, go to Manage > Licenses. 
# In the License portal, go to Manage > All Products. Select the License you want to assign to Users or Groups. For example, if 
# you have a Microsoft 365 Business Standard license or a Entra ID P1 license, select it. Click on Assign. Click on "Add users 
# and groups". Then select the specified Users or Groups.

# Devices refer to the physical devices used by users that access the Entra ID Tenant resources. So if you log into Entra 
# applications in your Tenant from your work computer inside the office, that will be represented as a Device. 

# a Managed Device is a device that has certain minimum standards, such as encryption, password requirements, anti-virus software 
# installed and running, among other security policies. In other words, you set organizational standards and define that only 
# devices that meet these standards can access applications within your Tenant, including Single sign-on (SSO) and Conditional 
# Access.

# Role-Based Access Control (RBAC) is a security feature in Azure that allows you to control access to Azure resources based 
# on a user's role within an organization (Entra ID Tenant). It enables you to grant access to Azure resources (e.g. Storage
#  Account, Virtual Machine, Azure Databases) to users, groups, or service principals (applications) based on their roles
# Azure RBAC does not directly control access to Entra ID resources, such as Users, Groups and Service Principals.

# Importantly, access to Entra ID resources, such as Users, Groups and Service Principals, is managed through Entra ID's built-in 
# roles and permissions, which are separate from Azure RBAC. Entra ID's built-in roles are considered "administrative roles". 
# They are accessible under Entra ID > Manage > Roles and administrators.

# So with Entra ID administrative roles, you can assign administrative roles to users, groups, or service principals to control 
# their access to Entra ID resources, such as: User management (e.g., creating, updating, deleting users), Group management 
# (e.g., creating, updating, deleting groups), Service principal management (e.g., creating, updating, deleting service 
# principals), Directory permissions (e.g., reading, writing directory data)

# While Entra ID's built-in roles and permissions provide access to Entra resources, Azure RBAC allows Entra ID's Users, Groups 
# and Service Principals to gain access to Azure Resources. In RBAC, there is a concept of a role. A RBAC role is essentially 
# a set of allowed operations that can be executed on a specific resource or a set of resources. Permissions are assigned to 
# roles rather than users. Entra ID Users, Groups and Service Principals can be assigned zero, one or many Azure RBAC roles.

# Entra ID Roles and Administrators access is available at Entra ID > Manage > Roles and Administrators. When dealing with Entra 
# ID Roles and Administrators, we are dealing with Entra ID related Roles (not Azure RBAC!). These Entra ID Roles are called 
# Administrative Roles. They deal with administrative permissions only, not permissions on Azure Resources and not permissions 
# in an Entra application. When we discuss Entra App Registrations and Entra Enterprise Applications, you'll find out how to 
# configure permissions on an Entra application.

# Entra Cloud Sync: Support for synchronizing to a Microsoft Entra tenant from a multi-forest disconnected Active Directory 
# forest environment: The common scenarios include merger & acquisition (where the acquired company's AD forests are isolated 
# from the parent company's AD forests), and companies that have historically had multiple AD forests.

# START ON App Registration > OAuth 2.0 Code (Access Grant) Flow


oauth2_schema = OAuth2PasswordBearer(tokenUrl='token')